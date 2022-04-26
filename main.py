from scipy import optimize
from torch.optim.optimizer import Optimizer, required
import torch
import numpy as np
import copy
from torch.distributions.categorical import Categorical
import torch.nn as nn
from torch.utils.data import TensorDataset
import logging
import matplotlib.pyplot as plt

class MO_agent(nn.Module):
    def __init__(self):
        super(MO_agent, self).__init__()
        self.modules_ = nn.Sequential(nn.Flatten(),
                                      nn.Linear(50, 128, bias=True),
                                      nn.Tanh(),
                                      nn.Linear(128, 128, bias=True),
                                      nn.Tanh(),
                                      nn.Linear(128, 10, bias=True),
                                      nn.Softmax(dim=1))

    def forward(self, x):
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return self.modules_(x)


class MSO(Optimizer):
    r"""Implements MSO Optimization.
    Args:

    Example:
        # >>> optimizer = MSO(model.parameters,cfg={'agent':METHOD,'memory_size':10})
    """

    def __init__(self, params, cfg):

        defaults = dict()
        super(MSO, self).__init__(params, defaults)
        self._params = self.param_groups[0]['params']
        self.subspace = []
        self.num_params = len(
            torch.nn.utils.parameters_to_vector(self._params))

        self.x0 = torch.nn.utils.parameters_to_vector(
            self._params).detach().cpu()
        self.w_k = 1
        self.cum_grad = (torch.nn.utils.parameters_to_vector(
            self._params) * 0.).detach().cpu()
        self.x_k = 0

        self.MO_agent = None
        self.RB_agent = False
        if cfg['agent'] == 'MO':
            # Load agent model (inference)
            self.MO_agent = MO_agent()
            PATH = './model/mo_agent_nn.ckpt'
            self.MO_agent.load_state_dict(torch.load(PATH))
        elif cfg['agent'] == 'RB':
            self.RB_agent = True
        self.memory_size = cfg['memory_size']
        self.alpha_hist_size = 4
        self.previous_steps = []
        self.alpha_hist = []

    def __setstate__(self, state):
        super(MSO, self).__setstate__(state)

    ############################################################################
    def get_subspace(self):
        # return ORTH subspace
        return [
            (self.x_k - self.x0) / (1e-10 + torch.norm((self.x_k - self.x0))),
            self.cum_grad / (1e-10 + torch.norm(self.cum_grad))]

    def update_subspace(self, new_func_grad, new_point):
        # Update ORTH subspace
        self.x_k = new_point.detach().cpu()
        self.w_k = 0.5 + np.sqrt(0.25 + self.w_k ** 2)
        self.cum_grad = self.cum_grad + self.w_k * new_func_grad.detach().cpu()

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_().detach().cpu()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1).detach().cpu()
            else:
                view = p.grad.data.view(-1).detach().cpu()
            views.append(view)
        return torch.cat(views, 0)

    def modify_directions(self, alpha):
        vec = torch.nn.utils.parameters_to_vector(self._params)
        tmp_device = vec.device
        vec = vec.detach().cpu()
        for j, v in enumerate(self.subspace):
            vec += alpha[j] * v
        torch.nn.utils.vector_to_parameters(vec.to(tmp_device), self._params)

    def add_grad2subspace(self, grad):
        grad = grad.detach().cpu()
        self.subspace.append((grad / (1e-10 + torch.norm(grad))))

    ############################################################################
    ############################################################################

    def step(self, loss=required):
        """Performs a single optimization step.
        Arguments:
            loss (callable, required): A closure that reevaluates the model and
            returns the loss. The closure performs zero_grad() at first
        """

        previous_point = torch.nn.utils.parameters_to_vector(self._params)
        loss(grad_comp=True)
        old_func_grad = self._gather_flat_grad()

        # Define ORTH subspace
        self.subspace = self.get_subspace()
        # Add gradient to the subspace
        self.add_grad2subspace(old_func_grad)
        # Add previous steps to the subspace
        self.subspace = self.previous_steps + self.subspace

        def func(alpha):
            # f(x + alpha*P)
            self.modify_directions(alpha)
            value = loss(grad_comp=False).item()
            self.modify_directions(-alpha)
            return value

        def func_alpha_grads(alpha):
            # grad_alpha f(x + alpha*P)
            self.modify_directions(alpha)
            self.zero_grad()
            loss(grad_comp=True)
            network_grads = self._gather_flat_grad()
            grads = []
            for m in self.subspace:
                g = m.dot(network_grads).item()
                grads.append(g)

            self.modify_directions(-alpha)
            return np.array(grads)

        #Performing Subspace Optimization
        alpha = np.zeros((len(self.subspace))).astype(float)
        _alpha = optimize.minimize(func,
                                   alpha,
                                   method='BFGS',
                                   jac=func_alpha_grads,
                                   callback=None,
                                   options={'maxiter': 20, 'gtol': 1e-5})

        self.modify_directions(_alpha.x.reshape(-1))
        new_point = torch.nn.utils.parameters_to_vector(self._params)
        self.zero_grad()
        loss(grad_comp=True)
        new_func_grad = self._gather_flat_grad()

        # Update ORTH Subspace
        self.update_subspace(new_func_grad, new_point)

        alpha = np.abs(_alpha.x.astype(np.float32)[:-3])
        # Subspace Removal Update
        if len(self.previous_steps) >= self.memory_size:
            if self.MO_agent is not None:
                # MO update
                state = torch.tensor(alpha).unsqueeze(0)
                # Prepare alpha history tensor
                # dimension = (alpha_hist_size+1) X memory_size
                hist = copy.deepcopy(self.alpha_hist)
                out = []
                for e in hist:
                    tmp = [0.] * (self.alpha_hist_size - len(e))
                    tmp.extend(e)
                    out.append(tmp)
                alpha_hist = torch.tensor(out).transpose(0, 1)
                state = torch.cat((alpha_hist, state))
                state = state.unsqueeze(0)
                # Activate stochastic agent
                decision_prob = self.MO_agent(state)
                distr = Categorical(decision_prob.squeeze())
                pg_action = distr.sample()
                # Remove direction from previous steps memory
                self.previous_steps.pop(pg_action)
                # Remove alpha step sizes history of removed direction
                self.alpha_hist.pop(pg_action)
                # Remove most recent alpha step size of removed direction
                alpha = np.delete(alpha, pg_action.item())
            elif self.RB_agent:
                # RB update
                self.previous_steps.pop(np.argmin(np.abs(_alpha.x[:-3])))
            else:
                # Legacy update (SESOP or ORTH)
                if len(self.previous_steps) > 0:
                    self.previous_steps.pop(0)

        # Update alpha history
        if self.MO_agent is not None and len(alpha) > 0:
            for i, a in enumerate(alpha):
                self.alpha_hist[i].append(a)
                if len(self.alpha_hist[i]) > self.alpha_hist_size:
                    self.alpha_hist[i].pop(0)

        # Add new step p_k
        if self.memory_size >0:
            self.previous_steps.append(((new_point - previous_point) / (
                1e-10 + torch.norm(new_point - previous_point))).detach().cpu())
        if self.MO_agent is not None:
            self.alpha_hist.append([])

        assert len(
            self.previous_steps) <= self.memory_size, 'Subspace size mismatch'

        return


def robust_linear_regression_objective(dim, n=100, sigma_noise=0.1):
    """This objective is the loss function of robust linear regression. The
    example is described in section 4.2 of the paper:

    Ke Li, Jitendra Malik, "Learning to Optimize", ICLR 2017.

    The optimization variable x is used for the weight vector and for the bias.
    w -    first dim-1 elements of x
    bias - last element of x
    """

    dim -= 1  # input to regressor has d-1 dimensions due to bias

    w_opt = np.random.randn(dim, 1)
    b_opt = np.random.randn()

    X = [np.random.randn(n // 4, dim) + np.random.randn(1, dim)
         for _ in range(4)]
    Y = [np.dot(X_i, w_opt) + b_opt + sigma_noise *
         np.random.randn(n // 4, 1) for X_i in X]
    X = torch.cat([torch.from_numpy(X_i).float() for X_i in X])
    Y = torch.cat([torch.from_numpy(Y_i).float() for Y_i in Y])

    X_train = X[:int(len(X)*0.85)]
    Y_train = Y[:int(len(X)*0.85)]
    X_test = X[int(len(X)*0.85):]
    Y_test = Y[int(len(X)*0.85):]

    train_ds = TensorDataset(X_train, Y_train)
    test_ds = TensorDataset(X_test, Y_test)

    def criterion(pred, y):
        out = y - pred
        return ((out ** 2) / (1 + out ** 2)).mean(dim=0)
    return criterion, train_ds, test_ds


if __name__ == '__main__':
    class Net(torch.nn.Module):
        def __init__(self, dim=100):
            super(Net, self).__init__()
            self.fc = torch.nn.Linear(dim-1, 1)

        def forward(self, x):
            x = self.fc(x)
            return x

    device = torch.device("cuda" if False else "cpu")
    seed = 100
    np.random.seed(seed)
    torch.manual_seed(seed)
    # MO and RB: Efficient Meta Subspace Optimization, by Y Choukroun and M Katz
    # SE (SESOP): Sequential subspace optimization method for large-scale unconstrained problems by G Narkiss and M Zibulesky
    # ORTH (SE with memory_size=0): Orth-method for smooth convex optimization, by A Nemirovski
    methods = ['MO', 'RB', 'SE']
    num_trials = 100
    num_epochs = {'MO': 1500, 'RB': 1500, 'SE': 1500}
    trial_seeds = seed + np.arange(num_trials)

    logging.basicConfig(filename='./results.txt', level=logging.DEBUG)
    convergence_res = {}
    for method in methods:
        logging.info(method)
        res = []
        for i, seed in enumerate(trial_seeds):
            np.random.seed(seed)
            torch.manual_seed(seed)
            print(f'Running method: {method}, seed: ({i}/{len(trial_seeds)})')
            criterion, train_ds, test_ds = robust_linear_regression_objective(
                100, n=100+19)
            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=100, shuffle=True)
            model = Net(100).to(device)
            optimizer = MSO(model.parameters(), cfg={
                            'agent': f'{method}', 'memory_size': 10})
            res_curr = []
            for ii in range(num_epochs[method]):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    def loss():
                        return criterion(model(x), y)

                    def objective(grad_comp=False):
                        if grad_comp:
                            optimizer.zero_grad()
                            res = loss()
                            res.backward()
                            return res
                        else:
                            with torch.no_grad():
                                return loss()
                    optimizer.step(objective)
                print(f'Epoch {ii} Loss: {loss().item()}')
                logging.debug(f'Epoch {ii} Loss: {loss().item()}')
                res_curr.append(loss().item())
            print(f'Final Loss: {loss().item()}\n')
            res.append(res_curr)
        convergence_res[method] = np.stack(res)
    print(convergence_res)
    ###
    fig = plt.figure()
    for k,v in convergence_res.items():
        plt.semilogy(v.mean(0),linewidth=3,label=k)
        plt.fill_between(np.arange(0, v.shape[1]),
                    v.mean(0) - v.std(0),v.mean(0) + v.std(0),alpha=0.2)
    plt.legend()
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Convergence')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.grid(True, which="both", ls="--")
    plt.ylim([1e-4,2])
    plt.savefig(f'convergence_plot_{num_trials}.png')
    plt.close()
