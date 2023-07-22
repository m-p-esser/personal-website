# Personal Website

Hello, this is my personal website build with [Hugo](https://gohugo.io/) and the extraordinary [Monochrome](https://themes.gohugo.io/themes/hugo-theme-monochrome/) theme.

Visit my website on http://marcphilippesser.com/. I write about Data Science topics in my [blog](http://marcphilippesser.com/blog/)

## Directory Structure of this Repo
For reference check: https://gohugo.io/getting-started/directory-structure/

## My requirements
- Publish Blog articles using Jupyter Notebooks
- Publish useful data visualizations


## My Workflow

### Publish Changes (in general)

#### Build static pages
- `hugo --environment production --minify` which will create the static pages in `public/`

#### Start Hugo Server
- `hugo server --environment production`
- `hugo server --disableFastRender`

### Writing Blog Posts

- Write the post (do the thing)
