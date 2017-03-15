# neuropredict

Automatic estimation of predictive power of commonly used neuroimaging features as well as user-defined features.

The aim of this python module would be to automatically assess the predictive power of commonly used neuroimaging features (such as resting-state connectivity, fractional anisotropy, subcortical volumes and cortical thickness features) automatically read from the processing of popular tools such as FSL, DTIstudio, AFNI and Freesurfer, and present a comprehensive report on a given dataset. It is mainly aimed (to lower or remove the barriers) at clinical users who would like to understand what features and brain regions are discriminative in their shiny new dataset before diving into the deep grey sea of feature extraction and optimization.

PS: It sounds similar (on the surface) to other software available, however it is aimed to lower the barriers even further, or remove them altogether! All the user would need to provide are commonly used features (such as a Freesurfer output directory) and obtain an easy to read report, along with well-packaged export of performance metrics (for sharing and posthoc comparison) on the predictive power of the features they are interested in.

## Example application

### Context

Imagine you have just acquired a wonderful new dataset with certain number of diseased patients and healthy controls. In the case of T1 mri analysis, you typically start by preprocessing it wih your favourite software (such as Freesurfer), which produces a ton of segmentations and statistics within them (such as their volumes and cortical thickness). Typical scenario would be to examine group differences (e.g. between controls and disease_one or between controls and other_disease), find the most discriminative variables and/or their brain regions and report how they relate to know cognitive or neuropsychological measures. This analysis and the resulting insights is necessary and informs us better of the dataset. However, that's not the fullest extent of the analysis one could perform, as association studies do not inform us of the predictive utility of the aforementioned discriminative variables or regions, which needs to independently investigated.

### Predictive analysis
 Conducting a machine learning study (to assess the predictive utility of different regions, features or methods) is not trivial. In the simplest case, it requires one to understand standard techniques, learn one or two toolboxes and do the complex programming necessary to interface their data with ML toolbox (even with the help of well-written packages like nilearn that are meant for neuroimaging analysis). In addition, in order to properly evaluate the performance, the user needs to have a good grasp of the best practices in machine learning. Even if the user could produce certain numbers out of a black-box toolboxes, some more programming is necessary to make sense of the results and procude necessary plots for publications.

### Report
 Neuropredict is here to remove those barriers and make your life easier!

 All you need to do is take care of preprocessing and produce quality controlled output through popular software, and neuropredict will produce a comprehensive report (see figures below) of distribtions of cross-validated performance, confusion matrices, analysis into misclassification and an intuitive comparison across multiple features.

  For example, if you have a dataset with 7 controls, 8 disease_one and 9 other_disease, all you need to do is produce a meta data file as shown below (noting a class label for each subject):

```
3071,controls
3069,controls
3064,controls
3063,controls
3057,controls
3055,controls
3053,controls
5004,disease_one
5074,disease_one
5077,disease_one
5093,disease_one
5095,disease_one
5000,disease_one
5001,disease_one
5002,disease_one
5003,disease_one
5000,other_disease
5006,other_disease
5013,other_disease
5014,other_disease
5016,other_disease
5018,other_disease
5019,other_disease
5021,other_disease
5022,other_disease
```


and `neuropredict` will produce the figures (and numbers in a CSV files) as shown here:

![composite](docs/composite_flyer.001.png)

The higher resoluation PDFs are included in the [docs](docs) folder.

I hope this user-friendly tool would help you get started on the predictive analysis you've been wanting to do for a while.

## Features currently supported
* Freesurfer
  * Subcortical volumes
  * Wholebrain Aseg stats
  
## Feature in development
* Freesurfer
  * cortical thickness
  * gray matter density
  * structural covariance
* Any nibabel-readable data

## Usage:

### command line options:

```
usage: neuropredict [-h] -m METADATAFILE -o OUTDIR [-p POSITIVECLASS]
                    [-f FSDIR] [-a ATLASID] [-u USERDIR] [-t TRAIN_PERC]
                    [-r NUM_REP_CV]

optional arguments:
  -h, --help            show this help message and exit
  -m METADATAFILE, --metadatafile METADATAFILE
                        Abs path to file containing metadata for subjects to
                        be included for analysis. At the minimum, each subject
                        should have an id per row followed by the class it
                        belongs to. E.g. sub001,control sub002,control
                        sub003,disease sub004,disease
  -o OUTDIR, --outdir OUTDIR
                        Output folder to store features and results.
  -p POSITIVECLASS, --positiveclass POSITIVECLASS
                        Name of the positive class (Alzheimers, MCI or
                        Parkinsons etc) to be used in calculation of area
                        under the ROC curve. Default: class appearning second
                        in order specified in metadata file.
  -f FSDIR, --fsdir FSDIR
                        Abs. path of SUBJECTS_DIR containing the finished runs
                        of Freesurfer parcellation
  -a ATLASID, --atlas ATLASID
                        Name of the atlas to use for visualization. Default:
                        fsaverage, if available.
  -u USERDIR, --userdir USERDIR
                        Abs. path to an user's own features.This contains a
                        separate folder for each subject (named after its ID
                        in the metadata file) containing a file called
                        features.txt with one number per line. All the
                        subjects must have the number of features (#lines in
                        file)
  -t TRAIN_PERC, --train_perc TRAIN_PERC
                        Percentage of the smallest class to be reserved for
                        training. Must be in the interval [0.01 0.99].If
                        sample size is sufficiently big, we recommend 0.5.If
                        sample size is small, or class imbalance is high,
                        choose 0.8.
  -r NUM_REP_CV, --num_rep_cv NUM_REP_CV
                        Number of repetitions of the repeated-holdout cross-
                        validation. The larger the number, the better the
                        estimates will be.
```

# Dependencies
 * numpy
 * scikit-learn