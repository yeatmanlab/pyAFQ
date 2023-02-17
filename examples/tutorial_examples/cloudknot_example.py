"""
==========================================
Using cloudknot to run pyAFQ on AWS batch:
==========================================
One of the purposes of ``pyAFQ`` is to analyze large-scale openly-available
datasets, such as those in the
`Human Connectome Project <https://www.humanconnectome.org/>`_.

To analyze these datasets, large amounts of compute are needed.
One way to gain access to massive computational power is by using
cloud computing. Here, we will demonstrate
how to use ``pyAFQ`` in the Amazon Web Services cloud.

We will rely on the `AWS Batch Service <https://aws.amazon.com/batch/>`_ ,
and we will submit work into AWS Batch using software that our group
developed called `Cloudknot <https://nrdg.github.io/cloudknot/>`_.
"""

# ##########################################################################
# # Import cloudknot and set the AWS region within which computations will take
# # place. Setting a region is important, because if the data that you are
# # analyzing is stored in `AWS S3 <https://aws.amazon.com/s3/>`_ in a
# # particular region, it is best to run the computation in that region as well.
# # That is because AWS charges for inter-region transfer of data.
# import cloudknot as ck
# ck.set_region('us-east-1')

# ##########################################################################
# # Define the function to use
# # --------------------------
# # ``Cloudknot`` uses the single program multiple data paradigm of computing.
# # This means that the same function will be run on multiple different inputs.
# # For example, a ``pyAFQ`` processing function run
# # on multiple different subjects in a dataset.
# # Below, we define the function that we will use. Notice that
# # ``Cloudknot`` functions include the import statements of the dependencies
# # used. This is necessary so that ``Cloudknot`` knows
# # what dependencies to install into AWS Batch to run this function.


# def afq_process_subject(subject):
#     # define a function that each job will run
#     # In this case, each process does a single subject
#     import s3fs
#     # all imports must be at the top of the function
#     # cloudknot installs the appropriate packages from pip
#     import AFQ.data.s3bids as afs
#     from AFQ.api.group import GroupAFQ
#     import AFQ.definitions.image as afm

#     # Download the given subject to your local machine from s3
#     # Can find subjects more easily if they are specified in a
#     # BIDS participants.tsv file, even if it is sparse
#     study_ixi = afs.S3BIDSStudy(
#         "my_study",
#         "my_study_bucket",
#         "my_study_prefix",
#         subjects=[subject],
#         use_participants_tsv=True,
#         anon=False)
#     study_ixi.download(
#         "local_bids_dir",
#         include_derivs=["pipeline_name"])

#     # you can optionally provide your own segmentation file
#     # in this case, we look for a file with suffix 'seg'
#     # in the 'pipeline_name' pipeline,
#     # and we consider all non-zero labels to be a part of the brain
#     brain_mask_definition = afm.LabelledImageFile(
#         suffix='seg', filters={'scope': 'pipeline_name'},
#         exclusive_labels=[0])

#     # define the api AFQ object
#     myafq = GroupAFQ(
#         "local_bids_dir",
#         preproc_pipeline="pipeline_name",
#         brain_mask_definition=brain_mask_definition,
#         viz_backend_spec='plotly',  # this will generate both interactive html and GIFs # noqa
#         scalars=["dki_fa", "dki_md"])

#     # export_all runs the entire pipeline and creates many useful derivates
#     myafq.export_all()

#     # upload the results to some location on s3
#     myafq.upload_to_s3(
#         s3fs.S3FileSystem(),
#         "my_study_bucket/my_study_prefix/derivatives/afq")


# ##########################################################################
# # Here we provide a list of subjects that we have selected to process
# # to randomly select 3 subjects without replacement, instead do:
# # subjects = [[1], [2], [3]]
# # see the docstring for S3BIDSStudy.__init__ for more information
# subjects = ["123456", "123457", "123458"]

# ##########################################################################
# # Defining a ``Knot`` instance
# # ---------------------------------
# # We instantiate a class instance of the :class:`ck.Knot` class.
# # This object will be used to run your jobs.
# # The object is instantiated with the `'AmazonS3FullAccess'` policy,
# # so that it can write the results
# # out to S3, into a bucket that you have write permissions on.
# # Setting the `bid_percentage` key-word makes AWS Batch use
# # `spot EC2 instances <https://aws.amazon.com/ec2/spot/>`_ for the
# # computation. This can result in substantial cost-savings, as spot compute
# # instances can cost much less than on-demand instances.
# # However, not that spot instances can also
# # be evicted, so if completing all of the work is very time-sensitive,
# # do not set this key-word argument. Using the `image_github_installs`
# # key-word argument will install pyAFQ from GitHub.
# # You can also specify other forks and branches to install from.
# knot = ck.Knot(
#     name='afq-process-subject-201009-0',
#     func=afq_process_subject,
#     base_image='python:3.8',
#     image_github_installs="https://github.com/yeatmanlab/pyAFQ.git",
#     pars_policies=('AmazonS3FullAccess',),
#     bid_percentage=100)

# ##########################################################################
# # Launching the computation
# # --------------------------------
# # The :meth:`map` method of the :class:`Knot object maps each of the inputs
# # provided as a sequence onto the function and executes the function on each
# # one of them in parallel.
# result_futures = knot.map(subjects)

# ##########################################################################
# # Once computations have started, you can call the following
# # function to view the progress of jobs::
# #
# #     knot.view_jobs()
# #
# # You can also view the status of a specific job::
# #
# #     knot.jobs[0].status


# ##########################################################################
# # When all jobs are finished, remember to use the :meth:`clobber` method to
# # destroy all of the AWS resources created by the :class:`Knot`
# result_futures.result()
# knot.clobber(clobber_pars=True, clobber_repo=True, clobber_image=True)

# ##########################################################################
# # In a second :class:`Knot` object, we use a function that takes the
# # resulting profiles of each subject and combines them into one csv file.


# def afq_combine_profiles(dummy_argument):
#     from AFQ.api import download_and_combine_afq_profiles
#     download_and_combine_afq_profiles(
#         "my_study_bucket", "my_study_prefix")


# knot2 = ck.Knot(
#     name='afq_combine_subjects-201009-0',
#     func=afq_combine_profiles,
#     base_image='python:3.8',
#     image_github_installs="https://github.com/yeatmanlab/pyAFQ.git",
#     pars_policies=('AmazonS3FullAccess',),
#     bid_percentage=100)

# ##########################################################################
# # This knot is called with a dummy argument, which is not used within the
# # function itself. The `job_type` key-word argument is used to signal to
# # ``Cloudknot`` that only one job is submitted rather than the default
# # array of jobs.
# result_futures2 = knot2.map(["dummy_argument"], job_type="independent")
# result_futures2.result()
# knot2.clobber(clobber_pars=True, clobber_repo=True, clobber_image=True)
