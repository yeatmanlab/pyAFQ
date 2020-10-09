# import cloudknot and set the correct region
import cloudknot as ck
ck.set_region('us-east-1')


def afq_process_subject(subject):
    # define a function that each job will run
    # In this case, each process does a single subject
    import logging
    import s3fs
    # all imports must be at the top of the function
    # cloudknot installs the appropriate packages from pip
    import AFQ.data as afqd
    import AFQ.api as api
    import AFQ.mask as afm

    # set logging level to your choice
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Download the given subject to your local machine from s3
    study_ixi = afqd.S3BIDSStudy(
        "my_study",
        "my_study_bucket",
        "my_study_prefix",
        subjects=[subject],
        anon=False)
    study_ixi.download(
        "local_bids_dir",
        include_derivs=["pipeline_name"])

    # you can optionally provide your own segmentation file
    # in this case, we look for a file with suffix 'seg'
    # in the 'pipeline_name' pipeline,
    # and we consider all non-zero labels to be a part of the brain
    brain_mask = afm.LabelledMaskFile(
        'seg', {'scope': 'pipeline_name'}, exclusive_labels=[0])

    # define the api AFQ object
    myafq = api.AFQ(
        local_bids_dir,
        dmriprep="pipeline_name",
        brain_mask=brain_mask,
        viz_backend='plotly',  # this will generate both interactive html and GIFs
        scalars=["dki_fa", "dki_md"])

    # export_all runs the entire pipeline and creates many useful derivates
    myafq.export_all()

    # upload the results to some location on s3
    myafq.upload_to_s3(
        s3fs.S3FileSystem(),
        f"my_study_bucket/my_study_prefix/derivatives/afq")


# here we provide a list of subjects that we have selected to process
# to randomly select 3 subjects without replacement, instead do:
# subjects = [[1], [2], [3]]
# see the docstring for S3BIDSStudy.__init__ for more information
subjects = [123456, 123457, 123458]

# define the knot to run your jobs on
# this not bids for access to ec2 resources,
# so its jobs are cheaper to run but may be evicted
# installs pyAFQ from github
knot = ck.Knot(
    name='afq_process_subject-201009-0',
    func=afq_process_subject,
    base_image='python:3.8',
    image_github_installs="https://github.com/yeatmanlab/pyAFQ.git",
    pars_policies=('AmazonS3FullAccess',),
    bid_percentage=100)

# launch a process for each subject
result_futures = knot.map(subjects)


def merge_results(future):
    # after all the subjects are done, we create another knot
    # which takes the resulting profiles of each subject
    # and combines them into one csv file
    # this can also be done locally
    # by using download_and_combine_afq_profiles in AFQ.api
    def afq_combine_profiles(dummy_argument):
        from AFQ.api import download_and_combine_afq_profiles
        download_and_combine_afq_profiles(
            "temp", "my_study_bucket", "my_study_prefix/derivatives/afq")

    knot2 = ck.Knot(
        name='afq_combine_subjects-201009-0',
        func=afq_combine_profiles,
        base_image='python:3.8',
        image_github_installs="https://github.com/yeatmanlab/pyAFQ.git",
        pars_policies=('AmazonS3FullAccess',),
        bid_percentage=100)

    futures = knot.map(["dummy_argument"], job_type="independent")
    futures.add_done_callback(knot2.clobber)


result_futures.add_done_callback(merge_results)

##########################################################################
# this function can be called repeatedly in a jupyter notebook
# to view the progress of jobs
# knot.view_jobs()

# you can also view the status of a specific job
# knot.jobs[0].status

# When all jobs are finished, remember to clobber the knot
# either using the aws console or this function in jupyter notebook:
# knot.clobber(clobber_pars=True, clobber_repo=True, clobber_image=True)
##########################################################################
