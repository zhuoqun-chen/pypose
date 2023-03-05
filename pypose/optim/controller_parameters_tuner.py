import torch
from torch.autograd.functional import jacobian

class ControllerParametersTuner():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def tune(self, dynamic_system, initial_state, ref_states, controller, parameters, parameters_tuning_set, tau, states_to_tune, func_get_state_error):
        states_to_tune = states_to_tune.double()
        states = []
        dxdparam_gradients = []

        system_state = torch.clone(initial_state)
        controller_parameters = parameters
        states.append(system_state)
        dxdparam_gradients.append(torch.zeros([len(initial_state), len(controller_parameters)]).double())

        for index, ref_state in enumerate(ref_states):
            controller_input = controller.get_control(parameters=controller_parameters, state=system_state, ref_state=ref_state, feed_forward_quantity=None)
            system_new_state = dynamic_system.state_transition(system_state, controller_input, tau)

            # calcuate the derivative
            dhdx_func = lambda state: controller.get_control(parameters = controller_parameters, state = state, ref_state = ref_state, feed_forward_quantity = None)
            dhdxk_tensor = torch.squeeze(jacobian(dhdx_func, system_state))

            dhdparam_func = lambda params: controller.get_control(parameters = params, state = system_state, ref_state = ref_state, feed_forward_quantity = None)
            dhdparam_tensor = torch.squeeze(jacobian(dhdparam_func, controller_parameters))

            dfdxk_func = lambda system_state: dynamic_system.state_transition(state = system_state, input = controller_input, t = tau)
            dfdxk_tensor = torch.squeeze(jacobian(dfdxk_func, system_state))

            dfduk_func = lambda inputs: dynamic_system.state_transition(state = system_state, input = inputs, t = tau)
            dfduk_tensor = torch.squeeze(jacobian(dfduk_func, controller_input))

            states.append(system_new_state)
            system_state = system_new_state

            last_gradient = dxdparam_gradients[-1]
            dxdparam_gradients.append(
              torch.mm(dfdxk_tensor + torch.mm(dfduk_tensor, dhdxk_tensor), last_gradient) + torch.mm(dfduk_tensor, dhdparam_tensor)
            )

        # accumulate the gradients
        gradient_sum = torch.zeros([len(parameters), 1]).double()
        for ref_state_index in range(0, len(ref_states)):
            state_error = func_get_state_error(states[ref_state_index + 1], ref_states[ref_state_index])
            state_error = torch.mm(torch.t(state_error), states_to_tune)
            gradient_sum += torch.t(2 * torch.mm(state_error, dxdparam_gradients[ref_state_index]))

        min_parameters = parameters_tuning_set[0]
        max_parameters = parameters_tuning_set[1]
        controller_parameters = torch.min(max_parameters, torch.max(min_parameters, parameters - self.learning_rate * gradient_sum))

        return controller_parameters