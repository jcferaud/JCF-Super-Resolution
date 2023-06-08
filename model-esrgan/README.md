
Hello,

The ESRGAN model .pth file, which originally had a size of 154472965 bytes, has been divided into eight parts: xaa, xab, xac, xad, xae, xaf, xag, and xah.

To reassemble the original .pth file on a Linux system, you can use the following command:

cat xa* > esrgan_generator34_56_19.pth
