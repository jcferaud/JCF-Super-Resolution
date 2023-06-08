# Super-Resolution
Super-Resolution Based on ESRGAN model

Hi,

# You are required to run the Python script "jcf-image4x.py" with the following three essential arguments:
• "-i": This specifies the path of the low-resolution image that needs enhancement.
• "-o": This designates where the enhanced high-resolution image will be saved.
• "-model": This is the location of the ESRGAN model after it has been trained.

# To test this tool on your linux box, you need to run the following commands:

# 1. clone the Git Code
$ git clone https://github.com/jcferaud/JCF-Super-Resolution.git

# 2. recreate the esrgan .pth file
$ cd JCF-Super-Resolution
$ cat ./model-esrgan/xa* > ./model-esrgan/esrgan_generator34_56_19.pth

# 3. run the following command to generate a high-resolution image from this directory:
$ python3 jcf-esrgan4x.py -i "./input-images/lr_bicubic_0853.png" -o "./output-images/hr_esrgan_0853.png" -model "./model-esrgan/esrgan_generator34_56_19.pth"

# Reminder:
- Ensure to reconstruct the file esrgan_generator34_56_19.pth before running the script (refer to readme.md in the model-esrgan directory for more details).
- Ensure that you have installed the needed libraries which includes torch and torchvision 

Best regards
