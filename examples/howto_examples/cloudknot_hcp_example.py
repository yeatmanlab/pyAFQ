"""
==========================
AFQ with HCP data
==========================
This example demonstrates how to use the AFQ API to analyze HCP data.
For this example to run properly, you will need to gain access to the HCP data.
This can be done by following this instructions on the webpage
`here <https://wiki.humanconnectome.org/display/PublicData/How+To+Connect+to+Connectome+Data+via+AWS>`_.
We will use the ``Cloudknot`` library to run our AFQ analysis in the AWS 
Batch service (see also 
`this example <http://yeatmanlab.github.io/pyAFQ/auto_examples/cloudknot_example.html>`_).
In the following we will use ``Cloudknot`` to run multiple
configurations of pyAFQ on the HCP dataset. Specifically, here we will run
pyAFQ with different tractography seeding strategies. 
"""

# ##########################################################################
# # Import cloudknot and set the correct region. The HCP data is stored in `us-east-1`, so it's best
# # to analyze it there.
# import configparser
# import itertools
# import cloudknot as ck
# import os.path as op

# ck.set_region('us-east-1')

# ##########################################################################
# # Define a function to run. This function allows us to pass in the subject ID for the subjects we would
# # like to analyze , as well as strategies for seeding tractography (different masks and/or different
# # numbers of seeds per voxel).


# def afq_process_subject(subject, seed_mask, n_seeds,
#                         aws_access_key, aws_secret_key):
#     # define a function that each job will run
#     # In this case, each process does a single subject
#     import logging
#     import s3fs
#     # all imports must be at the top of the function
#     # cloudknot installs the appropriate packages from pip
#     from AFQ.data.fetch import fetch_hcp
#     from AFQ.api.group import GroupAFQ
#     import AFQ.definitions.image as afm

#     # set logging level to your choice
#     logging.basicConfig(level=logging.INFO)
#     log = logging.getLogger(__name__)

#     # Download the given subject to the AWS Batch machine from s3
#     _, hcp_bids = fetch_hcp(
#         [subject],
#         profile_name=False,
#         study="HCP_1200",
#         aws_access_key_id=aws_access_key,
#         aws_secret_access_key=aws_secret_key)

#     # We make a new seed mask for each process based off of the
#     # seed_mask argument, which is a string.
#     # This is to avoid any complications with pickling the masks.
#     if seed_mask == "roi":
#         seed_mask_obj = afm.RoiImage()
#     elif seed_mask == "fa":
#         seed_mask_obj = afm.ScalarImage("dti_fa")
#     else:
#         seed_mask_obj = afm.FullImage()

#     # Determined if n_seeds is per voxel or not
#     if n_seeds > 3:
#         random_seeds = True
#     else:
#         random_seeds = False

#     # set the tracking_params based off our inputs
#     tracking_params = {
#         "seed_mask": seed_mask_obj,
#         "n_seeds": n_seeds,
#         "random_seeds": random_seeds}

#     # use segmentation file from HCP to get a brain mask,
#     # where everything not labelled 0 is considered a part of the brain
#     brain_mask_definition = afm.LabelledImageFile(
#         suffix='seg', filters={'scope': 'dmriprep'},
#         exclusive_labels=[0])

#     # define the api GroupAFQ object
#     myafq = GroupAFQ(
#         hcp_bids,
#         brain_mask_definition=brain_mask_definition,
#         tracking_params=tracking_params)

#     # export_all runs the entire pipeline and creates many useful derivates
#     myafq.export_all()

#     # upload the results to some location on s3
#     myafq.upload_to_s3(
#         s3fs.S3FileSystem(),
#         (f"my_study_bucket/my_study_prefix_{seed_mask}_{n_seeds}"
#          f"/derivatives/afq"))


# ##########################################################################
# # In this example, we will process the data from the following subjects
# subjects = ["103818", "105923", "111312"]

# ##########################################################################
# # We will test combinations of different conditions:
# # subjects, seed masks, and number of seeds
# seed_mask = ["fa", "roi"]
# n_seeds = [1, 2, 1000000, 2000000]

# ##########################################################################
# # The following function creates all the combinations of the above lists, such that every subject is
# # run with every mask and every number of seeds.
# args = list(itertools.product(subjects, seed_mask, n_seeds))

# ##########################################################################
# # We assume that the credentials for HCP usage are stored in the home directory in a
# # `~/.aws/credentials` file. This is where these credentials are stored if the AWS CLI is used to
# # configure the profile. We use the standard lib ``configparser`` library
# # to get the relevant hcp keys from there.
# CP = configparser.ConfigParser()
# CP.read_file(open(op.join(op.expanduser('~'), '.aws', 'credentials')))
# CP.sections()
# aws_access_key = CP.get('hcp', 'AWS_ACCESS_KEY_ID')
# aws_secret_key = CP.get('hcp', 'AWS_SECRET_ACCESS_KEY')

# ##########################################################################
# # The following function will attach your AWS keys to each list in a list of lists
# # We use this with each list being a list of arguments,
# # and we append the AWS keys to each list of arguments, so that we can pass
# # them into the function to be used on AWS Batch to download the data into the
# # AWS Batch machines.


# def attach_keys(list_of_arg_lists):
#     new_list_of_arg_lists = []
#     for args in list_of_arg_lists:
#         arg_ls = list(args)
#         arg_ls.extend([aws_access_key, aws_secret_key])
#         new_list_of_arg_lists.append(arg_ls)
#     return new_list_of_arg_lists


# ##########################################################################
# # This calls the function to attach the access keys to the argument list
# args = attach_keys(args)

# ##########################################################################
# # Define the :meth:`Knot` object to run your jobs on. See
# # `this example <http://yeatmanlab.github.io/pyAFQ/auto_examples/cloudknot_example.html>`_ for more
# # details about the arguments to the object.
# knot = ck.Knot(
#     name='afq-hcp-tractography-201110-0',
#     func=afq_process_subject,
#     base_image='python:3.8',
#     image_github_installs="https://github.com/yeatmanlab/pyAFQ.git",
#     pars_policies=('AmazonS3FullAccess',),
#     bid_percentage=100)

# ##########################################################################
# # This launches a process for each combination.
# # Because `starmap` is `True`, each list in `args` will be unfolded
# # and passed into `afq_process_subject` as arguments.
# result_futures = knot.map(args, starmap=True)

# ##########################################################################
# # The following function can be called repeatedly in a jupyter notebook
# # to view the progress of jobs::
# #
# #     knot.view_jobs()
# #
# # You can also view the status of a specific job::
# #
# #     knot.jobs[0].status

# ##########################################################################
# # When all jobs are finished, remember to clobber the knot to destroy all the resources that were
# # created in AWS.
# result_futures.result()  # waits for futures to resolve, not needed in notebook
# knot.clobber(clobber_pars=True, clobber_repo=True, clobber_image=True)

# ##########################################################################
# # We continue processing to create another knot which takes the resulting profiles of each
# # combination and combines them all into one csv file


# def afq_combine_profiles(seed_mask, n_seeds):
#     from AFQ.api import download_and_combine_afq_profiles
#     download_and_combine_afq_profiles(
#         "my_study_bucket", f"my_study_prefix_{seed_mask}_{n_seeds}")


# knot2 = ck.Knot(
#     name='afq_combine_subjects-201110-0',
#     func=afq_combine_profiles,
#     base_image='python:3.8',
#     image_github_installs="https://github.com/yeatmanlab/pyAFQ.git",
#     pars_policies=('AmazonS3FullAccess',),
#     bid_percentage=100)

# ##########################################################################
# # the arguments to this call to :meth:`map` are all the different configurations of pyAFQ that we ran
# seed_mask = ["fa", "roi"]
# n_seeds = [1, 2, 1000000, 2000000]
# args = list(itertools.product(seed_mask, n_seeds))

# result_futures2 = knot2.map(args, starmap=True)
# result_futures2.result()
# knot2.clobber(clobber_pars=True, clobber_repo=True, clobber_image=True)
