with tempfile.TemporaryDirectory() as tempdir:
    Path(f'{tempdir}/hip/input').mkdir(parents=True)
    Path(f'{tempdir}/hip/output').mkdir(parents=True)
    Path(f'{tempdir}/knee/input').mkdir(parents=True)
    Path(f'{tempdir}/knee/output').mkdir(parents=True)
    Path(f'{tempdir}/ankle/input').mkdir(parents=True)
    Path(f'{tempdir}/ankle/output').mkdir(parents=True)


    examination.hip.save_image(tempdir + '/hip/input/hip_0000.nii.gz')
    examination.knee.save_image(tempdir + '/knee/input/knee_0000.nii.gz')
    examination.ankle.save_image(tempdir + '/ankle/input/ankle_0000.nii.gz')

    docker_cmd = [
        'docker',
        'run',
        '--rm',
        '--runtime=nvidia',
        '--gpus', 'all',
        '--shm-size', '32G',
        # '--user', f'{os.getuid()}:{os.getgid()}',
        '--group-add', 'root',
        '-v', f'{tempdir}:/app/mnt:rw,Z',
        'swestfechtel/nnunet_torsion:latest'
    ]

    proc = subprocess.run(
        docker_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    if proc.returncode != 0:
        self.logger.error(f"Container exited with code {proc.returncode}. Logs: {proc.stdout.decode('utf-8')}")
        raise RuntimeError(f"Segmentation job failed for {examination.identifier} with exit code {proc.returncode}.")

    self.logger.info(f"Container exited with code {proc.returncode}.")
    self.logger.debug(f'Container logs: {proc.stdout.decode("utf-8")}')

    tmp = nib.load(tempdir + '/hip/output/hip.nii.gz')
    tmp = Segmentation.from_nibabel(tmp)
    tmp.transform_coordinate_system()
    examination.hip_mask = tmp

    tmp = nib.load(tempdir + '/knee/output/knee.nii.gz')
    tmp = Segmentation.from_nibabel(tmp)
    tmp.transform_coordinate_system()
    examination.knee_mask = tmp

    tmp = nib.load(tempdir + '/ankle/output/ankle.nii.gz')
    tmp = Segmentation.from_nibabel(tmp)
    tmp.transform_coordinate_system()
    examination.ankle_mask = tmp