import torch
from torch import nn
from torch.autograd.functional import jacobian


class ControllerParametersTuner(nn.Module):
    r"""
    This class is the general implementation of the controller parameters tuner based on
    the DiffTune paper.

    Given a dynamic system, a corresponding controller and reference states, assuming all
    these components are differentiable, it is possbile to find the most suitable
    parameters by constructing the loss function $\mathbf{L}$ and computing the gradient of
    $\mathbf{L}$ with respect to controller parameters.

    Note:
        For the parameter initial_state and parameters, please use row/column vector to
        represent these variables at the same time.

    .. math::
        \begin{align*}
            \mathbf{L} &= \sum_i^N ||x_i - \hat x_i||^2 \\
            \nabla L_{\theta} = \sum_i^N \frac{\partial L}{x_i} \frac{x_i}{\theta}
        \end{align*},
    """
    def __init__(self, learning_rate, penalty_coefficient, device):
        self.learning_rate = learning_rate
        self.device = device
        self.penalty_coefficient = penalty_coefficient

    def tune(self, dynamic_system, initial_state, ref_states, controller, parameters,
             parameters_tuning_set, tau, states_to_tune, func_get_state_error):
        r"""
        Args:
            dynamic_system (pypose.module.dynamics):
            initial_state (Tensor):
            ref_states ():
            controller (pypose.module.controller): Linear or nonlinear controller to control
            the dynamic system
            parameters (Tensor): Controller parameters
            parameters_tuning_set (Tensor): This set gives the minimum and the maximum
                value the parameters can reach
            tau: time interval in system
            states_to_tune (Tensor): choose which state needs to be considered in the
                loss function, usually only position is chosen.
            func_get_state_error (function): function has two inputs: system state and
                ref state. The function needs to be provided by users considering system
                state and ref state are not always in the same formation.
        """
        states_to_tune = states_to_tune.double()
        states = []
        inputs = []
        dxdparam_gradients = []
        dukdparam_gradients = []

        system_state = torch.clone(initial_state)
        controller_parameters = parameters
        states.append(system_state)
        dxdparam_gradients.append(
            torch.zeros(
                [len(initial_state), len(controller_parameters)], device=self.device)
            .double())

        for index, ref_state in enumerate(ref_states):
            controller_input = controller.get_control(parameters=controller_parameters, state=system_state, ref_state=ref_state, feed_forward_quantity=None)
            system_new_state = dynamic_system.state_transition(system_state, controller_input, tau)

            # calcuate the state derivative wrt. the parameters and the input derivative wrt. the parameters
            dhdx_func = lambda state: controller.get_control(parameters = controller_parameters,
                                                              state = state, ref_state = ref_state, feed_forward_quantity = None)
            dhdxk_tensor = torch.squeeze(jacobian(dhdx_func, system_state))

            dhdparam_func = lambda params: controller.get_control(parameters = params,
                                                                  state = system_state, ref_state = ref_state, feed_forward_quantity = None)
            dhdparam_tensor = torch.squeeze(jacobian(dhdparam_func, controller_parameters))

            dfdxk_func = lambda system_state: dynamic_system.state_transition(state = system_state,
                                                                              input = controller_input, t = tau)
            dfdxk_tensor = torch.squeeze(jacobian(dfdxk_func, system_state))

            dfduk_func = lambda inputs: dynamic_system.state_transition(state = system_state,
                                                                        input = inputs, t = tau)
            dfduk_tensor = torch.squeeze(jacobian(dfduk_func, controller_input))


            states.append(system_new_state)
            inputs.append(controller_input)
            system_state = system_new_state

            last_gradient = dxdparam_gradients[-1]

            dxdparam_gradients.append(
                torch.mm(dfdxk_tensor + torch.mm(dfduk_tensor, dhdxk_tensor), last_gradient) + torch.mm(dfduk_tensor, dhdparam_tensor)
            )

            dukdparam_gradients.append(
                torch.mm(dhdxk_tensor, last_gradient) + dhdparam_tensor
            )

        # accumulate the gradients
        gradient_sum = torch.zeros([len(parameters), 1], device=self.device).double()

        for ref_state_index in range(0, len(ref_states)):
            state_error = func_get_state_error(states[ref_state_index + 1], ref_states[ref_state_index])
            state_error = torch.atleast_2d(state_error)
            state_error = torch.mm(state_error, states_to_tune)
            gradient_sum += torch.t(2 * torch.mm(state_error, dxdparam_gradients[ref_state_index]))
            gradient_sum += self.penalty_coefficient \
                * torch.t(2 * torch.mm(torch.atleast_2d(inputs[ref_state_index]), dukdparam_gradients[ref_state_index]))

        gradient_sum = torch.squeeze(torch.t(gradient_sum))

        min_parameters = parameters_tuning_set[0]
        max_parameters = parameters_tuning_set[1]
        controller_parameters = torch.min(max_parameters, torch.max(min_parameters, parameters - self.learning_rate * gradient_sum))

        return controller_parameters
