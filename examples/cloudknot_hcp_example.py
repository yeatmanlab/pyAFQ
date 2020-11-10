"""
==============================
Using cloudknot to run pyAFQ with multiple configurations on AWS batch,
using the HCP dataset:
==============================

The following is an example of how to use cloudknot to run multiple
confiugrations of pyAFQ on the HCP dataset. Specifically, here we will run
pyAFQ with different tractography seeding strategies. 
"""

# import cloudknot and set the correct region
import configparser
import itertools
import cloudknot as ck
ck.set_region('us-east-1')


def afq_process_subject(subject, seed_mask, n_seeds,
                        aws_access_key, aws_secret_key):
    # define a function that each job will run
    # In this case, each process does a single subject
    import logging
    import s3fs
    # all imports must be at the top of the function
    # cloudknot installs the appropriate packages from pip
    from AFQ.data import fetch_hcp
    import AFQ.api as api
    import AFQ.mask as afm

    import numpy as np
    import os.path as op

    # set logging level to your choice
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Download the given subject to your local machine from s3
    _, hcp_bids = fetch_hcp(
        [subject],
        profile_name=False,
        study=f"HCP_1200",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key)

    # We make a new seed mask for each process based off of the
    # seed_mask argument, which is a string.
    # This is to avoid any complications with pickling the masks.
    if seed_mask == "roi":
        seed_mask_obj = afm.RoiMask()
    elif seed_mask == "fa":
        seed_mask_obj = afm.ScalarMask("dti_fa")
    else:
        seed_mask_obj = afm.FullMask()

    # Determined if n_seeds is per voxel or not
    if n_seeds > 3:
        random_seeds = True
    else:
        random_seeds = False

    # set the tracking_params based off our inputs
    tracking_params = {
        "seed_mask": seed_mask_obj,
        "n_seeds": n_seeds,
        "random_seeds": random_seeds}

    # use segmentation file from HCP to get a brain mask,
    # where everything not labelled 0 is considered a part of the brain
    brain_mask = afm.LabelledMaskFile(
        'seg', {'scope': 'dmriprep'}, exclusive_labels=[0])

    # define the api AFQ object
    myafq = api.AFQ(
        hcp_bids,
        brain_mask=brain_mask,
        tracking_params=tracking_params)

    # export_all runs the entire pipeline and creates many useful derivates
    myafq.export_all()

    # upload the results to some location on s3
    myafq.upload_to_s3(
        s3fs.S3FileSystem(),
        (f"my_study_bucket/my_study_prefix/derivatives_afq_"
        f"{seed_mask}_{n_seeds}"))


# here we provide a list of subjects that we have selected to process
subjects = [103818, 105923, 111312]

# here we construct lists of everything we want to test:
subjects = [str(i) for i in subjects]
seed_mask = ["fa", "roi"]
n_seeds = [1, 2, 1000000, 2000000]

# and here we mix the above lists, such that every subject is tried with
# every mask and every number of seeds
args = list(itertools.product(subjects, seed_mask, n_seeds))

# Use configparser to get the relevant hcp keys
# Requires an hcp entry in your ~/.aws/credentials file
CP = configparser.ConfigParser()
CP.read_file(open(op.join(op.expanduser('~'), '.aws', 'credentials')))
CP.sections()
aws_access_key = CP.get('hcp', 'AWS_ACCESS_KEY_ID')
aws_secret_key = CP.get('hcp', 'AWS_SECRET_ACCESS_KEY')

# This function will attach your keys to each list in a list of lists
# We use this with each list being a list of arguments,
# and we append the aws keys to each list of arguments.


def attach_keys(list_of_arg_lists):
    new_list_of_arg_lists = []
    for args in list_of_arg_lists:
        arg_ls = list(args)
        arg_ls.extend([aws_access_key, aws_secret_key])
        new_list_of_arg_lists.append(arg_ls)
    return new_list_of_arg_lists


# here we attach the access keys to the argument list
args = attach_keys(args)

# define the knot to run your jobs on
# this not bids for access to ec2 resources,
# so its jobs are cheaper to run but may be evicted
# installs pyAFQ from github
knot = ck.Knot(
    name='afq_hcp_tractography-201110-0',
    func=afq_process_subject,
    base_image='python:3.8',
    image_github_installs="https://github.com/yeatmanlab/pyAFQ.git",
    pars_policies=('AmazonS3FullAccess',),
    bid_percentage=100)

# launch a process for each combination
# Because starmap is True, each list in args will be unfolded
# and passed into afq_process_subject as arguments
result_futures = knot.map(args, starmap=True)

##########################################################################
# this function can be called repeatedly in a jupyter notebook
# to view the progress of jobs
# knot.view_jobs()

# you can also view the status of a specific job
# knot.jobs[0].status
##########################################################################

# When all jobs are finished, remember to clobber the knot
# either using the aws console or this function in jupyter notebook:
result_futures.result()  # waits for futures to resolve, not needed in notebook
knot.clobber(clobber_pars=True, clobber_repo=True, clobber_image=True)

# we create another knot which takes the resulting profiles of each combination
# and combines them into one csv file


def afq_combine_profiles(seed_mask, n_seeds):
    from AFQ.api import download_and_combine_afq_profiles
    download_and_combine_afq_profiles(
        "temp", "my_study_bucket",
        f"my_study_prefix/derivatives/afq_{seed_mask}_{n_seeds}")


knot2 = ck.Knot(
    name='afq_combine_subjects-201110-0',
    func=afq_combine_profiles,
    base_image='python:3.8',
    image_github_installs="https://github.com/yeatmanlab/pyAFQ.git",
    pars_policies=('AmazonS3FullAccess',),
    bid_percentage=100)

# the args here are all the different configurations of pyAFQ that we ran
seed_mask = ["fa", "roi"]
n_seeds = [1, 2, 1000000, 2000000]
args = list(itertools.product(seed_mask, n_seeds))

result_futures2 = knot2.map(args, starmap=True)
result_futures2.result()
knot2.clobber(clobber_pars=True, clobber_repo=True, clobber_image=True)
