import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import os

def on_key(event):
    global key_pressed
    key_pressed = event.key

def plot_images(image_folder, file1, file2, title):
    # Construct the full paths for the images using the image folder and filenames
    img1_path = os.path.join(image_folder, file1)
    img2_path = os.path.join(image_folder, file2)

    # Read images from files
    img1 = mpimg.imread(img1_path)
    img2 = mpimg.imread(img2_path)

    # Create a figure with two subplots for the images
    plt.figure(figsize=(10, 5))

    # Subplot for the first image
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.axis('off')  # Turn off axis labels
    plt.title('File 1')

    # Subplot for the second image
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.axis('off')  # Turn off axis labels
    plt.title('File 2')

    # Set the overall title for the plot
    plt.suptitle(title)

    # Connect the key press event to the on_key function
    plt.gcf().canvas.mpl_connect('key_press_event', on_key)

    # Display the plot and wait for a key press or mouse button click
    while True:
        plt.waitforbuttonpress()

        global key_pressed
        key = key_pressed if 'key_pressed' in globals() else None

        # Handle the key press
        if key == 'right':
            plt.close()  # Close the current plot before opening the next one
            return 'right'
        elif key == 'left':
            plt.close()  # Close the current plot before opening the previous one
            return 'left'
        elif key == 'escape':
            plt.close('all')  # Close all open plots to exit the program
            return 'escape'

def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Read and plot images from a CSV file.')
    parser.add_argument('--csv_file', type=str, required=True, help='CSV file containing rows of file1, file2, number.')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing the images.')
    parser.add_argument('--reverse', action='store_false', help='Sort rows by the third column in ascending order.')

    args = parser.parse_args()

    # Read rows from the CSV file
    with open(args.csv_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        # Sort rows by the float value of the third column
        rows.sort(key=lambda x: float(x[2]), reverse=args.reverse)

        total_rows = len(rows)
        current_row = 0

        while current_row < total_rows:
            # Extract information from the current row
            file1, file2, number = rows[current_row]

            # Plot images based on the current row and image folder
            key = plot_images(args.image_folder, file1, file2, number)

            # Increment or decrement the row index after displaying the plot
            if key == 'right':
                current_row += 1
            elif key == 'left':
                current_row -= 1
            elif key == 'escape':
                break  # Exit the loop and end the program

            # Ensure the row index stays within bounds
            current_row = max(0, min(current_row, total_rows - 1))

if __name__ == "__main__":
    main()
