Modeling white matter
=====================

In order to make inferences about white matter tissue properties, we use a
variety of models. The models are fit to the data in each voxel and the
parameters of the model are used to interpret the signal.

For an interesting perspective on modeling of tissue properties from diffusion
MRI data, please refer to a recent paper by Novikov and colleagues
[Novikov2018]_.

`This page <https://yeatmanlab.github.io/pyAFQ/reference/methods.rst>` includes
a list of the model parameters that are accessible through the
:class:`AFQ.api.group.GroupAFQ` and :class:`AFQ.api.participant.ParticipantAFQ`
objects.

.. [Novikov2018] Novikov DS, Kiselev VG, Jespersen SN. On modeling. Magn Reson
    Med. 2018 Jun;79(6):3172-3193. doi: 10.1002/mrm.27101. Epub 2018 Mar 1.
    PMID: 29493816; PMCID: PMC5905348.