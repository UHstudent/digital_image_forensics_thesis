# Master’s Thesis Engineering
I believe this work will be easy to digest and can be a good starting point for anyone wishing to learn more about Digital Image Forensics, start analysing images for signs of manipulation, or wishes to start their own project on Digital Image Forensics.

### Title - Digital Image Forensics: A quantitative & qualitative comparison between State-of-the-art-AI and Traditional Techniques for detection and localization of image manipulations
Thesis text: https://github.com/UHstudent/digital_image_forensics_thesis/blob/main/Thesis%20text_Digital%20Image%20Forensics-A%20Comparative%20Study%20between%20AI%20and%20traditional%20approaches.pdf

#### This directory contains all code referenced in my thesis text. Feel free to use any code for your own purposes.

## Summary
- Open source Python implementation of three traditional techniques: JPEG Ghosts, Resampling, Noise wavelet blocking
- Guide on how to effectively use these techniques to identify manipulations
- Graphical UI contributions to Sherloq, an open source digital image forensic toolset: https://github.com/GuidoBartoli/sherloq
- Study of recent works in the Field of Digital image forensics and their implications
- Comparison of a modern (2023) state-of-the-art AI (MM-Fusion: https://github.com/IDT-ITI/MMFusion-IML) and three traditional techniques for the detection and localization of manipulations
- Critical discussion of the use of AI in the field of Digital Image Forensics
- Future vision for the place and use of AI solutions in Digital Image Forensics
- As of writing (2024), traditional techniques continue to be relevant for those wanting to reliably identify manipulation in images

## Code in this library
- Development scripts for each traditional technique: These scripts are ideal for understanding how the principles of the original papers have been translated into code. These scripts are also an ideal starting point for making improvements or adapting the techniques for your own purposes.
- dataset_evaluation_pipeline_scripts: These scripts were used to evaluate the datasets used in this work, and subsequently process the result files to make ROC curves and calculate overlap statistics,... Datasets featured in this work: IMD2020, Columbia, In the Wild, CocoGlide, IFS training set, Coverage
- Sherloq implementations: UI implementations as they were originally submitted to Sherloq, together with an adapted development script for completeness. The current tools in Sherloq may deviate depending on community changes.

# Thank you
If you have read my work, found it usefull or have any questions, it would be an honor to hear from you. Please open an Issue on this Github page for this purpose, that would make me happy
