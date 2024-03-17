# Distill-diffusion-TBD-
This is the repository that includes the distillation outcome of the DDPM(CIFAR10), which has the 8, 32, 125 steps process respectively.

## Outcome Detail

![Image Description](assets/distill_combined_image_8.png)

## Training Detail

If u wanna see the detail of the loss during the training process and the intermediate generated results, I have upload the log to the Wandb.

Using the Link to access to my Wandb project. [Distillation_to_time_step8](https://wandb.ai/2623448751/pg_dis_train_and_sample_student_same_teacher/runs/8fi5h7ew?nw=nwuser2623448751) I put the whole training process in one figure to compare the loss.

So every time a certain number of epoches is reached, the model will be updated and distillation will start again. At this time, the loss will rise very high. With the training goes on, then the loss would decrease.

## Model ckpt

The model .ckpt has been uploaded to the huggingface.

Using the Link to access to my model. [Aragonaa/Distill_Ddpm_CIFAR10](https://huggingface.co/Aragonaa/Distill_Ddpm_CIFAR10/tree/main) 

## Inference

*1 Downloading the models from the link in Model ckpt*

*2 Then git clone the repository and install the dependencied* 

    git clone https://github.com/digbangbang/Distill-diffusion-TBD-.git

    pip install -r requirements.txt

*3 Put the .ckpt same as the [Space in Huggiing Face](https://huggingface.co/spaces/Aragonaa/distill_ddpm_CIFAR10/tree/main)*

*4 Let's sampling!*

Just change the `time_step` in the model .ckpt that u have downloaded, then u can sample with the distilled model.
    
    python sample.py --time_step 8

If u wanna try the original pretrained model in the same `time_step`, then using following command.

    python sample.py --time_step 8 --use_pretrained

*5 Find the results in sample document*

## Other methods and models are TBD...
