# Super-Resolution
Super-Resolution Based on ESRGAN model


Hello,

# Please follow these instructions to execute the Python script "jcf-esrgan4x.py" using the mandatory command-line arguments below:

"-i" denotes the path to the low-resolution image for enhancement.

"-o" signifies the location where the enhanced high-resolution image will be saved.

"-model" is the path to the trained ESRGAN model.

# To use this tool on your Linux environment, execute the following commands:

1. Clone the GitHub repository:

   $ git clone https://github.com/jcferaud/JCF-Super-Resolution.git

2. Recreate the ESRGAN .pth file by navigating to the JCF-Super-Resolution directory and running:

   $ cd JCF-Super-Resolution

   $ cat ./model-esrgan/xa* > ./model-esrgan/esrgan_generator35_56_45.pth


3. Run the command below from this directory to generate a high-resolution image (this is a sample command):

   $ python3 ./jcf-esrgan4x.py -i "./input-images/lr_bicubic_0853.png" -o "./output-images/hr_esrgan_0853.png" -model "./model-esrgan/esrgan_generator35_56_45.pth"

# Important Reminders:

Make sure to rebuild the esrgan_generator35_56_45.pth file before running the above command. More details can be found in the readme.md file in the model-esrgan directory.

Make sure also to install the required libraries listed in the requirements.txt file.

Best regards.
