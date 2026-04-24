from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import  BaseOutput
import torch
import math
from dataclasses import dataclass
@dataclass
class SchedulerOutput(BaseOutput):
    """
    Base class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor

class Modified_DPMSolverMultistepScheduler(DPMSolverMultistepScheduler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas = None,
        solver_order: int = 2,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        template_scale = 5,
        **kwargs,
    ):
        super(Modified_DPMSolverMultistepScheduler, self).__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            beta_schedule,
            trained_betas,
            solver_order,
            prediction_type,
            thresholding,
            dynamic_thresholding_ratio,
            sample_max_value,
            algorithm_type,
            solver_type,
            lower_order_final,
            **kwargs,
        )
        self.template_c = 3
        print("----------------template_scale-----------:" ,template_scale )
        self.template_scale = template_scale
    def create_template_points(self, theta , lengths):
        axis = []
        for length in lengths:
            length = int(32 *length)
            for theta_ in theta:
                axis.append(  [32 + int( length*math.cos( math.radians( theta_) ) ), 32 + int( length*math.sin(  math.radians(theta_ ))) ]  )
        return axis
    
    def add_peak( self , inputs ,x , y ):
        inputs_modify = inputs.clone() 
        # mean = inputs[ 0 , self.template_c , x - 5: x+5 , y - 5: y+5].mean()
        # std = inputs[ 0 ,  self.template_c , x - 5: x+5 , y - 5: y+5].std()
        mean = inputs.mean()
        std = inputs.std()
        inputs_modify[0 , self.template_c ,  x - 1: x+1 , y - 1: y+1 ] = 0
        inputs_modify[0 , self.template_c , x , y] = mean + self.template_scale*std

        return inputs_modify

    def peak_injection(self, inputs , theta , lengths):
        points = self.create_template_points(theta , lengths)
        z_fft = torch.fft.fftshift(torch.fft.fft2(inputs.float() ), dim=(-1, -2))
        for i in range(len(points)):
            z_fft = self.add_peak(z_fft, points[i][0] ,points[i][1] )
        init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(z_fft, dim=(-1, -2))).real
        return init_latents_w.half()
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ):
        """
        Step function propagating the sample with the multistep DPM-Solver.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        step_index = (self.timesteps == timestep).nonzero()
        if len(step_index) == 0:
            step_index = len(self.timesteps) - 1
        else:
            step_index = step_index.item()
        prev_timestep = 0 if step_index == len(self.timesteps) - 1 else self.timesteps[step_index + 1]
        lower_order_final = (
            (step_index == len(self.timesteps) - 1) and self.config.lower_order_final and len(self.timesteps) < 15
        )
        lower_order_second = (
            (step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, timestep, sample)
        #### model output = -> x_0 
        # print('template injection')
        model_output = self.peak_injection(model_output , [1 , 135 ] , [ 0.2 ,0.3 ,0.4 ,0.5])

        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(model_output, timestep, prev_timestep, sample)
        elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            timestep_list = [self.timesteps[step_index - 1], timestep]
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs, timestep_list, prev_timestep, sample
            )
        else:
            timestep_list = [self.timesteps[step_index - 2], self.timesteps[step_index - 1], timestep]
            prev_sample = self.multistep_dpm_solver_third_order_update(
                self.model_outputs, timestep_list, prev_timestep, sample
            )

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)
    def convert_model_output(
        self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Convert the model output to the corresponding type that the algorithm (DPM-Solver / DPM-Solver++) needs.

        DPM-Solver is designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to
        discretize an integral of the data prediction model. So we need to first convert the model output to the
        corresponding type to match the algorithm.

        Note that the algorithm type and the model type is decoupled. That is to say, we can use either DPM-Solver or
        DPM-Solver++ for both noise prediction model and data prediction model.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the converted model output.
        """
        # DPM-Solver++ needs to solve an integral of the data prediction model.
        if self.config.algorithm_type == "dpmsolver++":
            if self.config.prediction_type == "epsilon":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.config.prediction_type == "sample":
                x0_pred = model_output
            elif self.config.prediction_type == "v_prediction":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                # Dynamic thresholding in https://arxiv.org/abs/2205.11487
                orig_dtype = x0_pred.dtype
                if orig_dtype not in [torch.float, torch.double]:
                    x0_pred = x0_pred.float()
                dynamic_max_val = torch.quantile(
                    torch.abs(x0_pred).reshape((x0_pred.shape[0], -1)), self.config.dynamic_thresholding_ratio, dim=1
                )
                dynamic_max_val = torch.maximum(
                    dynamic_max_val,
                    self.config.sample_max_value * torch.ones_like(dynamic_max_val).to(dynamic_max_val.device),
                )[(...,) + (None,) * (x0_pred.ndim - 1)]
                x0_pred = torch.clamp(x0_pred, -dynamic_max_val, dynamic_max_val) / dynamic_max_val
                x0_pred = x0_pred.type(orig_dtype)
                
            return x0_pred
        # DPM-Solver needs to solve an integral of the noise prediction model.
        elif self.config.algorithm_type == "dpmsolver":
            if self.config.prediction_type == "epsilon":
                return model_output
            elif self.config.prediction_type == "sample":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                epsilon = (sample - alpha_t * model_output) / sigma_t
                return epsilon
            elif self.config.prediction_type == "v_prediction":
                alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
                epsilon = alpha_t * model_output + sigma_t * sample
                return epsilon
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverMultistepScheduler."
                )
