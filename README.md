# msc_fiona

This repository contains a directory structure with analyses scripts to be used by Fiona for her MSc thesis.

The directory structure is the following

msc_fiona
  - scripts
    - modules
    - evaluation
    - variability_analysis
    - extreme_analysis
  - plots
  - manuscript


## Instructions

Clone this repository to your local machine and create your own branch. 

    git checkout -b [name_of_your_new_branch, e.g. "fiona"]

Push this new branch to the github repository using

    git push origin [name_of_your_new_branch, e.g. "fiona"]

List all the available branches using

    git branch -a

To change between branches use

    git checkout [name_of_branch]

Make sure that you are always in your branch when you commit your changes.

Stage your created/changed files for a commit 

    git add file.py

    git commit -m "Commit message..."

    git push

To push the current branch and set the remote as upstream (required for the first time), use

    git push --set-upstream origin [name_of_your_new_branch, e.g. "fiona"]

## Things to discuss/explain/introduce

*Scientific elements*:
  - ROMSOC model setup
    - ROMS (ocean)
    - COSMO (atmosphere)
    - OASIS (coupler)
  - simulation procedure
    - spinup/hindcast
    - PGW (future projection approach)
  - Open scientific questions
    - increase in extremes
    - acidification extremes

*Data Analysis setup*:
  - using the terminal
  - getting access to Servers
  - setting up remote connections using SSH
  - Using python
    - using VSCode as IDE
    - *.py versus *.ipynb
  - introducing different work environments (sea/kryo/meso/euler)
    - home directory (50GB) - good place to create local clone of github repository
    - data storage under /nfs/sea/work/ 

*Manuscript writing*:
  - using overleaf (LaTeX)
  - can be linked to a GitHub repository for version control and as an additional backup in case overleaf fails

*Data Analysis scripts*:
  - the scripts are used for data analysis (i.e., to create the results/plots/tables)
  - analysis should contain a model evaluation (how well does the model reproduce observations (mean, variability)?)
    - using observational products such as Glodap, CalCOFI, OSPapa, Argo?, see e.g., Desmet et al. (2022,2023)
  - analysis should then focus on acidification extremes
    - extreme analysis usually involves:
      - calculating a variable climatology and a (seasonally varying) extreme thresold 
        - this is obviously not required if an absolute threshold is used
        - thinking about long term trends brings up the question of using a fixed vs. moving baseline
      - identifying when/where the acidity exceeds the threshold (generating a boolean array)
      - potentially connecting extreme grid cells across space and time with a unique identifier (labeling)
      - extracting grid cell indices/coordinates and calculate extreme characteristics (e.g., duration, mean intensity, severity, ...)
      - project extreme characteristics back onto a map or onto a timeline
  - investigating the variability in certain parameters can provide additional insights next to the extreme analysis 
  - **CHECK IN SCRIPTS DIRECTORY FOR AVAILABLE EXAMPLES TO START FROM**

*What type of extreme analysis should be conducted?*:
  - multiple carbonate chemistry paramteres exist: 
    - H+ ions (pH)
    - Omega_Aragonite (Omega_Calcite)
    - DIC
    - Alk
  - investigating H+ ion concentration extremes using a seasonally varying 95th percentile at the surface and at depth (in a fixed and a moving basline)
  - often the saturation horizon (shallowest depth where Omega_Aragonite = 1) is regarded
    - saturation horizons have been calculated for Omega_Aragonite = 1, Omega_Aragonite = 1.3, Omega_Aragonite = 1.5, Omega_Aragonite = 1.7
  - extreme variability of the depth of the saturation horizons would be of great interest
  - **NEED TO IDENTIFY A CLEAR RESEARCH QUESTION (incl. HYPOTHESIS)** 


