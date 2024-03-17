# Distill-diffusion-TBD-
This is the repository that includes the distillation outcome of the DDPM(CIFAR10), which has the 8, 32, 125 steps process respectively.

## Outcome Detail



## Training Detail

If u wanna see the detail of the loss during the training process and the intermediate generated results, I have upload the log to the Wandb.

Usiing the Link to access to my Wandb project. [Distillation_to_time_step8](https://wandb.ai/2623448751/pg_dis_train_and_sample_student_same_teacher/runs/8fi5h7ew?nw=nwuser2623448751) I put the whole training process in one figure to compare the loss.

So every time a certain number of epoches is reached, the model will be updated and distillation will start again. At this time, the loss will rise very high. With the training goes on, then the loss would decrease.

## Model ckpt


