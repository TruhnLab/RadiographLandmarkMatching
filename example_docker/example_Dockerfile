FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set the working directory
WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip

RUN pip3 install "numpy==1.26.4"
RUN pip3 install "torch==2.2.1" --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install "nnunetv2==2.5.1"

RUN mkdir -p /app/nnUNet_results/Dataset008_TorsionHipTrainVal/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all
RUN mkdir -p /app/nnUNet_results/Dataset021_TorsionKneeTrainVal/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all
RUN mkdir -p /app/nnUNet_results/Dataset022_TorsionAnkleTrainVal/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all

RUN mkdir -p /app/mnt

COPY ./checkpoints/hip/fold_all/checkpoint_best.pth /app/nnUNet_results/Dataset008_TorsionHipTrainVal/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/
COPY ./checkpoints/knee/fold_all/checkpoint_best.pth /app/nnUNet_results/Dataset021_TorsionKneeTrainVal/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/
COPY ./checkpoints/ankle/fold_all/checkpoint_best.pth /app/nnUNet_results/Dataset022_TorsionAnkleTrainVal/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/

COPY ./checkpoints/hip/plans.json /app/nnUNet_results/Dataset008_TorsionHipTrainVal/nnUNetTrainer__nnUNetPlans__3d_fullres/
COPY ./checkpoints/knee/plans.json /app/nnUNet_results/Dataset021_TorsionKneeTrainVal/nnUNetTrainer__nnUNetPlans__3d_fullres/
COPY ./checkpoints/ankle/plans.json /app/nnUNet_results/Dataset022_TorsionAnkleTrainVal/nnUNetTrainer__nnUNetPlans__3d_fullres/

COPY ./checkpoints/hip/dataset.json /app/nnUNet_results/Dataset008_TorsionHipTrainVal/nnUNetTrainer__nnUNetPlans__3d_fullres/
COPY ./checkpoints/knee/dataset.json /app/nnUNet_results/Dataset021_TorsionKneeTrainVal/nnUNetTrainer__nnUNetPlans__3d_fullres/
COPY ./checkpoints/ankle/dataset.json /app/nnUNet_results/Dataset022_TorsionAnkleTrainVal/nnUNetTrainer__nnUNetPlans__3d_fullres/

COPY ./checkpoints/hip/dataset_fingerprint.json /app/nnUNet_results/Dataset008_TorsionHipTrainVal/nnUNetTrainer__nnUNetPlans__3d_fullres/
COPY ./checkpoints/knee/dataset_fingerprint.json /app/nnUNet_results/Dataset021_TorsionKneeTrainVal/nnUNetTrainer__nnUNetPlans__3d_fullres/
COPY ./checkpoints/ankle/dataset_fingerprint.json /app/nnUNet_results/Dataset022_TorsionAnkleTrainVal/nnUNetTrainer__nnUNetPlans__3d_fullres/

COPY ./entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

RUN chown -R root:root /app && chmod -R 775 /app

ENTRYPOINT ["/app/entrypoint.sh"]