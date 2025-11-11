# Code for Analyzing Key Point Generation Methods on Different Datasets

To execute the code, first navigate to the **`scripts`** folder and run the corresponding **sbatch scripts**.  
These scripts include all necessary **dependencies** required to run the Python files within **Docker containers**.

If the **default values** are sufficient, you only need to modify the **input** and **output paths** in the sbatch scripts.

If something doesn’t work, it’s recommended to check the code manually — especially the **default values** and **path configurations**, as these are often the cause of most errors.  
Also, ensure that the **models are properly loaded**. Normally, the models are included in the repository to guarantee that the exact same versions are used.  
However, if the repository was **freshly cloned**, the models may **not yet be available locally**.  
In that case, run the **`download_models`** script to retrieve the external models.

Model locations can be chosen freely — just make sure that the **paths in the scripts** correctly point to the models you intend to use.

---

## Running the Method Code

In the folder **`scripts/KPG`**, you’ll find the Slurm files used to execute the different methods.  
Update the **input** and **output paths** accordingly.  
If necessary, double-check that the **model paths** in the script match your chosen locations.

Example:

```bash
sbatch khosravani2024.slurm
```

## Running the Evaluation

In the folder scripts/evaluation, you’ll find the sbatch files used to perform evaluations.
The most relevant scripts are:
	•	evaluation_referencefree – runs the reference-free evaluation framework described in the paper.
	•	evaluation_reproduction – runs the standard metrics from the original publications to compare your results with those reported by the authors.

```bash
sbatch evaluation_referencefree.slurm
sbatch evaluation_reproduction.slurm
```

## Modifying Results

The folder **`scripts/modified_KPG`** contains the **pre- and post-processing scripts** for the KPG methods.

- **`data-processing.slurm`**  
  Runs the **quality model** on all arguments to evaluate and score them.

- **`from_900_to_3x300.slurm`**  
  Used when you decide to select the **900 highest-quality arguments** and split them into **three blocks of 300** each.  
  This allows the arguments to be processed normally by the KPG methods in smaller, manageable batches.

- **`reduce_keypoints_from_multiple_topics.slurm`**  
  Takes the **3×300 keypoint sets** and removes all **duplicates**.  
  This script serves as the **post-processing step** for the *n = 900* setting, ensuring a unique and consolidated set of keypoints.
  

