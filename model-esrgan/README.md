
Hello,

The ESRGAN model .pth file has been divided into eight parts: xaa, xab, xac, xad, xae, xaf, xag, and xah.

To reassemble the original .pth file on a Linux system, you can use the following command:

cat xa* > esrgan_generator35_56_45.pth
