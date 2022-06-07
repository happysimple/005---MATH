# -- Project configuration -----------------------------------------------------
project = 'Math'
author = '王开心'
copyright = '2022, 王开心'
release = '1.0.0'

# -- General configuration ------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

extensions = [
    'myst_parser',            # Markdown
    'sphinx.ext.mathjax',     # Math
    'sphinx_copybutton',      # Copy Code
    'sphinx.ext.githubpages', # Gighub
    'sphinx_panels',          # panels
]


# -- Options for HTML output ----------------------------------------------
templates_path = ['_templates']
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    "prev_next_buttons_location": "bottom",
}


