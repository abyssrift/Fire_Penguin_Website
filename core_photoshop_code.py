from PIL import Image
import math
import numpy as np
from scipy.ndimage import gaussian_filter
import random as rand
from flask import Flask,render_template,request

# Open an image file
image = Image.open("new_rose.jpg")


def grayscale (local_image):
    print("Currently Attempting to apply Grayscale Filter...")
    pixels = local_image.load()
    for i in range(local_image.width):
        for j in range(local_image.height):
            r, g, b = pixels[i,j]
            average = int((r + g + b)/3)
            pixels[i,j] = (average, average, average)
    print("Filter applied successfully, exiting filter function")        
    return local_image

def lovelyPink(local_image):
    print("Currently Attempting to apply lovelypink filter...")
    pixels = local_image.load()
    for i in range(local_image.width):
        for j in range(local_image.height):
            r, g, b = pixels[i,j]
            newRed = 0.393 * r + 0.768 * g + 0.189 * b
            newRed = min(int(newRed),255)
            newBlue = 0.349 * r + 0.686 * g + 0.168 * b
            newBlue = min(int(newBlue),255)
            newGreen = 0.272 * r + 0.534 * g + 0.131 * b
            newGreen = min(int(newGreen),255)
            pixels[i,j] = (newRed,newGreen,newBlue)
    return(local_image)

def vintageSepia(local_image):
    print("Currently Attempting to apply sepia filter...")
    pixels = local_image.load()
    for i in range(local_image.width):
        for j in range(local_image.height):
            R, G, B = pixels[i,j]
            new_R = int(0.393 * R + 0.769 * G + 0.189 * B)
            new_G = int(0.349 * R + 0.686 * G + 0.168 * B)
            new_B = int(0.272 * R + 0.534 * G + 0.131 * B)

            new_R = min(new_R, 255)
            new_G = min(new_G, 255)
            new_B = min(new_B, 255)
            pixels[i,j] = (new_R,new_G,new_B)
    return(local_image)

def Echo(local_image):
    print("Currently Attempting to apply sepia filter...")
    pixels = local_image.load()
    for i in range(local_image.width):
        for j in range(local_image.height):
            R, G, B = pixels[i,j]
            new_R = int(0.272 * R + 0.534 * G + 0.131 * B)
            new_G = int(0.349 * R + 0.686 * G + 0.168 * B)
            new_B = int(0.393 * R + 0.769 * G + 0.189 * B)

            new_R = min(new_R, 255)
            new_G = min(new_G, 255)
            new_B = min(new_B, 255)
            pixels[i,j] = (new_R,new_G,new_B)
    return(local_image)

def Vibrance(local_image):
    print("Currently Attempting to apply sepia filter...")
    pixels = local_image.load()
    for i in range(local_image.width):
        for j in range(local_image.height):
            R, G, B = pixels[i,j]
            new_R = int(0.272 * R + 0.734 * G + 0.131 * B)
            new_G = int(0.349 * R + 0.986 * G + 0.168 * B)
            new_B = int(0.193 * R + 0.769 * G + 0.189 * B)

            new_R = min(new_R, 255)
            new_G = min(new_G, 255)
            new_B = min(new_B, 255)
            pixels[i,j] = (new_R,new_G,new_B)
    return(local_image)


def vignettefocus(local_image):
    print("Currently Attempting to apply vignette filter...")
    pixels = local_image.load()
    center_x = local_image.width // 2
    center_y = local_image.height // 2
    max_distance = math.sqrt(center_x**2 + center_y**2)

    for y in range(local_image.height):
        for x in range(local_image.width):
            dx = x - center_x
            dy = y - center_y
            distance = math.sqrt(dx**2 + dy**2)

            darkening_factor = 1 - (distance / max_distance) ** 0.95
            R, G, B = pixels[x, y]

            new_R = int(R * darkening_factor)
            new_G = int(G * darkening_factor)
            new_B = int(B * darkening_factor)
            pixels[x, y] = (new_R, new_G, new_B)
    return local_image

def contrast(local_image):
    print("Currently Attempting to apply contrast filter...")
    pixels = local_image.load()
    for i in range(local_image.width):
        for j in range(local_image.height):
            R,G,B = pixels[i,j]
            intensity = (R+G+B)//3
            if(intensity > 160):
                R +=40
                G +=40
                B +=40
                R = min(255,R)
                G = min(255,G)
                B = min(255,B)
                pixels[i,j] = (R,G,B)
            if(intensity < 100):
                R -=40
                G -=40
                B -=40
                R = max(0,R)
                G = max(0,G)
                B = max(0,B)
                pixels[i,j] = (R,G,B)
    return(local_image)


def bluredges(local_image):
    print("Currently Attempting to apply blur filter...")
    pixels = local_image.load()
    center_x = local_image.width // 2
    center_y = local_image.height // 2
    width = local_image.width
    height = local_image.height
    kernel = 0
    max_distance = math.sqrt(center_x**2 + center_y**2)
    for i in range(width):
        for j in range(height):
            dx = i - center_x
            dy = j - center_y
            distance = math.sqrt(dx**2 + dy**2)

            kernel = int((distance / max_distance) * 5)
            r_sum = 0
            g_sum = 0
            b_sum = 0
            count = 0
            for x in range(-kernel,kernel + 1):
                for y in range (-kernel,kernel + 1):
                    if((i + x >= 0 and i + x < width) and (j + y >= 0 and j + y < height)):
                        R,G,B = pixels[i+x,j+y]
                        r_sum += R
                        g_sum += G
                        b_sum += B
                        count +=1
            if(count > 0):
                avg_R = r_sum // count
                avg_G = g_sum // count
                avg_B = b_sum // count

            # Set the pixel to the average color
            pixels[i, j] = (avg_R, avg_G, avg_B)
            
    return (local_image)
         

def fast_gaussian_blur(local_image, sigma=3):
    """
    Applies a fast Gaussian blur to an image using scipy's gaussian_filter, which is optimized
    and performs the separable filter internally.

    Parameters:
    local_image (PIL.Image): The input image to blur.
    sigma (float): The standard deviation for Gaussian kernel (controls blur amount).

    Returns:
    PIL.Image: The blurred image.
    """
    print("Currently attempting to apply Gaussian blur filter...")

    # Convert image to numpy array
    pixels = np.array(local_image)

    # Apply Gaussian blur with the given sigma value
    blurred_pixels = gaussian_filter(pixels, sigma=(sigma, sigma, 0))

    # Convert back to PIL image
    result_image = Image.fromarray(blurred_pixels.astype('uint8'))
    return result_image

def lighten(local_image):
    print("Currently Attempting to apply lighten filter...")
    pixels = local_image.load()
    for i in range(local_image.width):
        for j in range(local_image.height):
            R, G, B = pixels[i,j]
            new_R = R + 40
            new_G = G + 40
            new_B = B + 40

            new_R = min(new_R, 255)
            new_G = min(new_G, 255)
            new_B = min(new_B, 255)
            pixels[i,j] = (new_R,new_G,new_B)
    return(local_image)

def darken(local_image):
    print("Currently Attempting to apply darken filter...")
    pixels = local_image.load()
    for i in range(local_image.width):
        for j in range(local_image.height):
            R, G, B = pixels[i,j]
            new_R = R - 40
            new_G = G - 40
            new_B = B - 40

            new_R = max(new_R, 0)
            new_G = max(new_G, 0)
            new_B = max(new_B, 0)
            pixels[i,j] = (new_R,new_G,new_B)
    return(local_image)

def infrared(local_image):
    print("Currently Attempting to apply infrared filter...")
    pixels = local_image.load()
    for i in range(local_image.width):
        for j in range(local_image.height):
            R,G,B = pixels[i,j]
            G = G-255
            B = B-255
            pixels[i,j] = (R,G,B)
    return(local_image)

def noise(local_image):
    print("Currently Attempting to apply noise filter...")
    pixels = local_image.load()
    for i in range(local_image.width):
        for j in range(local_image.height):
            R, G, B = pixels[i,j]
            value1 = rand.randint(1,20)
            value2 = rand.randint(1,2)
            if(value2 == 1):
                new_R = R  - value1
                new_G = G  - value1
                new_B = B  - value1
                new_R = max(new_R, 0)
                new_G = max(new_G, 0)
                new_B = max(new_B, 0)
            elif (value2 == 2):
                new_R = R  + value1
                new_G = G  + value1
                new_B = B  + value1
                new_R = min(new_R, 255)
                new_G = min(new_G, 255)
                new_B = min(new_B, 255)
            pixels[i,j] = (new_R,new_G,new_B)
    return(local_image)

def colorpopRed(local_image):
    print("Currently Attempting to apply color pop filter...")
    pixels = local_image.load()

    for i in range(local_image.width):
        for j in range(local_image.height):
            R, G, B = pixels[i, j]
            
            # Define the criteria for pixels to keep in color
            if R > 80 and G < 150 and B < 150:
                continue
            else:
                # Convert to grayscale if not meeting the criteria
                avg = (R + G + B) // 3
                pixels[i, j] = (avg, avg, avg)

    return local_image

image = colorpopRed(image)

image.save("colorpopredtest2.jpg")






