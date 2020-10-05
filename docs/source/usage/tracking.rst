Usage tracking with Google Analytics
------------------------------------

To be able to assess usage of the software, we are recording each use of the
CLI as an event in Google Analytics, using `popylar <https://popylar.github.io>`_.

The only information that we are recording is the fact that the CLI was called.
In addition, through Google Analytics, we will have access to very general
information, such as the country and city in which the computer using the CLI
was located and the time that it was used. At this time, we do not record any
additional information, although in the future we may want to record statistics
on the computational environment in which the CLI was called, such as the
operating system.

Opting out of this usage tracking can be done by calling the CLI with the
`--notrack` flag::

    pyAFQ /path/to/config.toml --notrack