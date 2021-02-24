"""
Streamline count viewer
==========================
Generate a brainsprite viewer for a streamline coun map with an anatomical
background, by populating an html template.
"""

from brainsprite import viewer_substitute
import tempita

anat = './callosal_tract_profile/dti_FA.nii.gz'
stat_img = './callosal_tract_profile/afq_AntFrontal_density_map.nii.gz'

bsprite = viewer_substitute(vmin=0, cmap="hot", symmetric_cmap=False,
                            title="streamline_count")
bsprite.fit(stat_img, bg_img=anat)

template = \
"""
<!DOCTYPE html>
<html lang="en">
<body>
  <!-- This is the div that will host the html brainsprite code -->
  <div id="div_viewer">
    <!-- Note the curly brackets are here to tell tempita to update this element. -->
    {{ html }}
  </div>

  <!-- Import jquery and brainsprite javascript libraries -->
  <script src="https://code.jquery.com/jquery-3.5.0.min.js"></script>
  <script>
    // We'll inject brainsprite.js here, to make the html self-contained.
    {{ bsprite }}
  </script>

  <!-- That's where the js brainsprite code will live -->
  <script>
  // On load: build all figures
  $( window ).on('load',function() {
    // Create brain slices
    var brain = brainsprite(
      // that's where all the brainsprite parameters will be injected.
      {{ js }}
    );
  });
  </script>
</body>
</html>
"""

template = tempita.Template(template)

viewer = bsprite.transform(template, javascript='js', html='html', library='bsprite')
output_fname = 'plot_stat_map.html'
viewer.save_as_html(output_fname)

viewer