# Super-Resolution
Super-Resolution Based on ESRGAN model

Hi,

You are required to run the Python script "jcf-image4x.py" with the following three essential arguments:\n
• "-i": This specifies the path of the low-resolution image that needs enhancement.\n
• "-o": This designates where the enhanced high-resolution image will be saved.\n
• "-model": This is the location of the ESRGAN model after it has been trained.\n
\n
Here's an example of how to execute the command to generate a high-resolution image from this directory:\n

$ python3 jcf-esrganx4.py -i "./input-images/lr_bicubic_0853.png" -o "./output-images/hr_esrgan_0853.png" -model "./model-esrgan/esrgan_generator34_56_19.pth"\n
\n
Reminder:\n
Ensure to reconstruct the file esrgan_generator34_56_19.pth before running the script (refer to readme.md in the model-esrgan directory for more details).\n
\n
Best regards
