import torch
import torch.nn.functional as F
import math


class NoiseScheduleVP:
    def __init__(self, schedule='linear'):

        if schedule not in ['linear', 'cosine']:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be 'linear' or 'cosine'".format(schedule))
        self.beta_0 = 0.1
        self.beta_1 = 20
        self.cosine_s = 0.008
        self.cosine_beta_max = 999.
        self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (
                    1. + self.cosine_s) / math.pi - self.cosine_s
        self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
        self.schedule = schedule
        if schedule == 'cosine':
            # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
            # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
            self.T = 0.9946
        else:
            self.T = 1.

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t
        else:
            raise ValueError("Unsupported ")

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0 ** 2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        else:
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (
                        1. + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            return t


def model_wrapper(model, noise_schedule=None, is_cond_classifier=False, classifier_fn=None, classifier_scale=1.,
                  time_input_type='1', total_N=1000, model_kwargs={}):


    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        """
        if time_input_type == '0':
            # discrete_type == '0' means that the model is continuous-time model.
            # For continuous-time DPMs, the continuous time equals to the discrete time.
            return t_continuous
        elif time_input_type == '1':
            # Type-1 discrete label, as detailed in the Appendix of DPM-Solver.
            return 1000. * torch.max(t_continuous - 1. / total_N, torch.zeros_like(t_continuous).to(t_continuous))
        elif time_input_type == '2':
            # Type-2 discrete label, as detailed in the Appendix of DPM-Solver.
            max_N = (total_N - 1) / total_N * 1000.
            return max_N * t_continuous
        else:
            raise ValueError("Unsupported time input type {}, must be '0' or '1' or '2'".format(time_input_type))

    def cond_fn(x, t_discrete, y):
        """
        Compute the gradient of the classifier, multiplied with the sclae of the classifier guidance.
        """
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier_fn(x_in, t_discrete)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return classifier_scale * torch.autograd.grad(selected.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if is_cond_classifier:
            y = model_kwargs.get("y", None)
            if y is None:
                raise ValueError("For classifier guidance, the label y has to be in the input.")
            t_discrete = get_model_input_time(t_continuous)
            noise_uncond = model(x, t_discrete, **model_kwargs)
            noise_uncond = noise_uncond.sample if not isinstance(noise_uncond, torch.Tensor) else noise_uncond
            cond_grad = cond_fn(x, t_discrete, y)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = len(cond_grad.shape) - 1
            return noise_uncond - sigma_t[(...,) + (None,) * dims] * cond_grad
        else:
            t_discrete = get_model_input_time(t_continuous)
            model_output = model(x, t_discrete, **model_kwargs)
            model_output = model_output.sample if not isinstance(model_output, torch.Tensor) else model_output
            return model_output

    return model_fn


class DPM_Solver:
    def __init__(self, model_fn, noise_schedule):

        self.model_fn = model_fn
        self.noise_schedule = noise_schedule

    def get_time_steps(self, skip_type, t_T, t_0, N, device):

        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T, lambda_0, N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t = torch.linspace(t_0, t_T, 10000000).to(device)
            quadratic_t = torch.sqrt(t)
            quadratic_steps = torch.linspace(quadratic_t[0], quadratic_t[-1], N + 1).to(device)
            return torch.flip(
                torch.cat([t[torch.searchsorted(quadratic_t, quadratic_steps)[:-1]], t_T * torch.ones((1,)).to(device)],
                          dim=0), dims=[0])
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_time_steps_for_dpm_solver_fast(self, t_T, t_0, steps, device):

        K = steps // 3 + 1
        if steps % 3 == 0:
            orders = [3, ] * (K - 2) + [2, 1]
        elif steps % 3 == 1:
            orders = [3, ] * (K - 1) + [1]
        else:
            orders = [3, ] * (K - 1) + [2]
        timesteps = self.get_time_steps('logSNR', t_T, t_0, K, device)
        return orders, timesteps

    def dpm_solver_first_update(self, x, s, t, return_noise=False):

        ns = self.noise_schedule
        dims = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_t = ns.marginal_std(t)

        phi_1 = torch.expm1(h)

        noise_s = self.model_fn(x, s)
        x_t = (
                torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,) * dims] * x
                - (sigma_t * phi_1)[(...,) + (None,) * dims] * noise_s
        )
        if return_noise:
            return x_t, {'noise_s': noise_s}
        else:
            return x_t

    def dpm_solver_second_update(self, x, s, t, r1=0.5, noise_s=None, return_noise=False):

        ns = self.noise_schedule
        dims = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(
            s1), ns.marginal_log_mean_coeff(t)
        sigma_s1, sigma_t = ns.marginal_std(s1), ns.marginal_std(t)

        phi_11 = torch.expm1(r1 * h)
        phi_1 = torch.expm1(h)

        if noise_s is None:
            noise_s = self.model_fn(x, s)
        x_s1 = (
                torch.exp(log_alpha_s1 - log_alpha_s)[(...,) + (None,) * dims] * x
                - (sigma_s1 * phi_11)[(...,) + (None,) * dims] * noise_s
        )
        noise_s1 = self.model_fn(x_s1, s1)
        x_t = (
                torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,) * dims] * x
                - (sigma_t * phi_1)[(...,) + (None,) * dims] * noise_s
                - (0.5 / r1) * (sigma_t * phi_1)[(...,) + (None,) * dims] * (noise_s1 - noise_s)
        )
        if return_noise:
            return x_t, {'noise_s': noise_s, 'noise_s1': noise_s1}
        else:
            return x_t

    def dpm_solver_third_update(self, x, s, t, r1=1. / 3., r2=2. / 3., noise_s=None, noise_s1=None, noise_s2=None):

        ns = self.noise_schedule
        dims = len(x.shape) - 1
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(
            s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)

        phi_11 = torch.expm1(r1 * h)
        phi_12 = torch.expm1(r2 * h)
        phi_1 = torch.expm1(h)
        phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.
        phi_2 = torch.expm1(h) / h - 1.

        if noise_s is None:
            noise_s = self.model_fn(x, s)
        if noise_s1 is None:
            x_s1 = (
                    torch.exp(log_alpha_s1 - log_alpha_s)[(...,) + (None,) * dims] * x
                    - (sigma_s1 * phi_11)[(...,) + (None,) * dims] * noise_s
            )
            noise_s1 = self.model_fn(x_s1, s1)
        if noise_s2 is None:
            x_s2 = (
                    torch.exp(log_alpha_s2 - log_alpha_s)[(...,) + (None,) * dims] * x
                    - (sigma_s2 * phi_12)[(...,) + (None,) * dims] * noise_s
                    - r2 / r1 * (sigma_s2 * phi_22)[(...,) + (None,) * dims] * (noise_s1 - noise_s)
            )
            noise_s2 = self.model_fn(x_s2, s2)
        x_t = (
                torch.exp(log_alpha_t - log_alpha_s)[(...,) + (None,) * dims] * x
                - (sigma_t * phi_1)[(...,) + (None,) * dims] * noise_s
                - (1. / r2) * (sigma_t * phi_2)[(...,) + (None,) * dims] * (noise_s2 - noise_s)
        )
        return x_t

    def dpm_solver_update(self, x, s, t, order):

        if order == 1:
            return self.dpm_solver_first_update(x, s, t)
        elif order == 2:
            return self.dpm_solver_second_update(x, s, t)
        elif order == 3:
            return self.dpm_solver_third_update(x, s, t)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5):

        ns = self.noise_schedule
        s = t_T * torch.ones((x.shape[0],)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_noise=True)
            higher_update = lambda x, s, t, **kwargs: self.dpm_solver_second_update(x, s, t, r1=r1, **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.dpm_solver_second_update(x, s, t, r1=r1, return_noise=True)
            higher_update = lambda x, s, t, **kwargs: self.dpm_solver_third_update(x, s, t, r1=r1, r2=r2, **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1. / order).float(), lambda_0 - lambda_s)
            nfe += order
        print('adaptive solver nfe', nfe)
        return x

    def sample(self, x, steps=10, eps=1e-4, T=None, order=3, skip_type='logSNR',
               adaptive_step_size=False, fast_version=True, atol=0.0078, rtol=0.05,
               ):

        t_0 = eps
        t_T = self.noise_schedule.T if T is None else T
        device = x.device
        if adaptive_step_size:
            with torch.no_grad():
                x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol)
        else:
            if fast_version:
                orders, timesteps = self.get_time_steps_for_dpm_solver_fast(t_T=t_T, t_0=t_0, steps=steps,
                                                                            device=device)
            else:
                N_steps = steps // order
                orders = [order, ] * N_steps
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=N_steps, device=device)
            with torch.no_grad():
                for i, order in enumerate(orders):
                    vec_s, vec_t = torch.ones((x.shape[0],)).to(device) * timesteps[i], torch.ones((x.shape[0],)).to(
                        device) * timesteps[i + 1]
                    x = self.dpm_solver_update(x, vec_s, vec_t, order)
        return x