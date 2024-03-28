## Background

Image quality is an important characteristic of millimeter wave (MMW) imaging systems, such as those widely deployed in US airports.
The IEEE N42.59 standard [1] describes test objects, test methods, and objective analysis algorithms for measuring several aspects of image quality.  
This code implements the lateral resolution test method [2]. 
MMW systems within the scope of IEEE N42.59 use a variety of imaging approaches and produce both 2D and 3D images.
We use the term _lateral resolution_ to refer to spatial resolution in the plane perpecficular to the synthetic aperture.
The method for measuring spatial resolution in the depth direction, perpendicular to the lateral plane, is described in [3]. 

## Instructions

Data analysis steps:
1) Collect at least 5 images of one or both of the lateral resolution test objects.
1) Crop each image down to the edges of the test object (see figure above).
Place all the cropped images into a folder e.g. 'test_folder2/'
1) Run the process_whole_lat_res_folder() function in "process_whole_lat_res_folder.py".
The code near the end of that file can be used to change images are searched for.

```
if __name__== "__main__" :
    folder = "test_folder2/"
    file_str = '*.tif'
    process_lat_res_folder(folder, file_str, PDF_obj=True)
```

4) Setting PDF_obj to True means all the figures are output to a PDF file called "N4259_lateral_resolution_results_summary.pdf"
in the folder "test_folder2/". Find that file and interpret your results!

## Software description

This is version 0.7 of this code and was published in March 2023.
This is an alpha release and future improvements are expected.
Please let us know if you find bugs or have suggestions for improvements.
This code was developed and is maintained by Dr Jack L. Glover, Radiation Physics Division, National Institute of Standards and Technology (NIST).
Feel free to email him at firstname dot lastname at agency dot gov.


## Acknowledgement

If you use this code then please cite the paper:
- **Title**: A standardized test for measuring the spatial resolution of active millimeter wave imaging systems 
- **Author**: Jack L. Glover
- Intended to be published in: Proceedings of the SPIE _Radar Sensor Technology XXVIII_

## References 

- [1] IEEE N42.59: _Standard for Measuring the Imaging Performance of Active Millimeter-Wave Systems for Security Screening of Humans_. https://standards.ieee.org/ieee/N42.59/11515/
- [2] Jack L. Glover. _A standardized test for measuring the spatial resolution of active millimeter wave imaging systems_. Published in the proceedings of the Radar Sensor Technology XXVIII conference. http://spie.org/DCS207
- [3] David M. Sheen, R. Trevor Clark, and Maurio Grando. _Depth resolution evaluation for an active wideband millimeter-wave imaging system_. Passive and Active Millimeter-Wave Imaging XXIV. Vol. 11745. SPIE, 2021. [link](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11745/117450M/Depth-resolution-evaluation-for-an-active-wideband-millimeter-wave-imaging/10.1117/12.2587201.full)

