# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = pyAFQ
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

upload: html
		./upload-gh-pages.sh build/html/ pyAFQ yeatmanlab

.PHONY: distclean realclean

# find artifacts from sphinx-gallery in examples folder
ARTIFACTS := $(strip \
			 	$(wildcard ../examples/*.gz) $(wildcard ../examples/**/*.gz) \
			 	$(wildcard ../examples/*.trk) $(wildcard ../examples/**/*.trk) \
			 	$(wildcard ../examples/*.npy) $(wildcard ../examples/**/*.npy) \
			  )

# leaves only the files that were in the distribution. deletes files created
# by sphinx-build within source
distclean: clean
	@echo Removing files created by sphinx-build
	rm -rf $(BUILDDIR)
	rm -f $(SOURCEDIR)/usage/config.rst
	rm -rf $(SOURCEDIR)/auto_examples/
	$(if $(ARTIFACTS),rm -f $(ARTIFACTS))

# delete more files than distclean. this would also remove the data files.
realclean: distclean
	@echo Removing data files from $(HOME)
	rm -rf $(HOME)/.dipy/
	rm -rf $(HOME)/AFQ_data/