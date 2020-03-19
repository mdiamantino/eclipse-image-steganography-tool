<img src="eclipse/resources/eclipse_icon.png" align="right" width="100"/>


# Eclipse

Eclipse is a **steganography command-line tool** (and module) that can be used to hide and extract messages from images.

## Goal
*Eclipse* aims to be an **efficient tool** for image steganography, combining different concepts of computer science, such as **cryptography**, **compression** and **machine learning**.

## How it works
In contrast to secure communication, where an encrypted message is exchanged publicly, the main goal of steganography is to hide the message, so is difficult to detect the transfer and in worst case scenario, even more challenging retrieving the original message. Hence the exchange goes **unnoticed**.

![Eclipse Diagram](eclipse/resources/eclipse_diagram.png)

### Detailed description
In order to achieve a reasonable efficiency and security level, *Eclipse* focuses on three fundamentals:
- **Avoiding image comparison** at  all costs:
	- **Why:** If the *stego-image* (the one containing the hidden message) can be compared to the *cover image* (the one without), differences could encourage inspections.
	- **How:**  *Eclipse* avoids, as far as possible, that the cover-image using two tools of **machine learning**:
		- **Image Augmentation**: The message is never hidden in the original image. Instead, *Eclipse* performs **random transformations** on it so that it could be really hard to generate the same image again and compare it to the stego-image;
		- **Black-box adversarial attack** *[feature under development]*: Image recognition tools could be used to search for the original image (and eventually proceed to brute-force the above-mentioned point). *Eclipse* aims to perform black-box adversarial attacks, so most of image recognition systems would misidentify the possible stego-image.
		- **Metadata suppression**:*Eclipse* deletes all EXIFS, in order to discourage image traceability such as **GPS coordinates**, **origin**, **author** and so forth.
- **Minimizing differences** between the two images:
	- **Why:**  If *Eclipse* fails the first point and the original cover-image is found,  then it is crucial that differences are negligible and imperceptible, so that they could be easily associated to eventual transfer/compression operations.
	- **How:**  
		- *Eclipse* steganography technique is based on **discrete cosine transform**, which compared to the standard LSB technique, is **far more robust, safe and imperceptible**. In addition, the technique has been modified so that the message is hidden only **one bit/highest coefficient** -in order to reduce machine perceptibility- and exclusively in the **Cb is blue minus luma** component -which is the worst noticed by the human eye.
		- The message is **randomly and uniformly distributed** in the image.
- **Message encryption**:
	- **Why**: When the first to points fail, the original message needs to be unreadable.
	- **How**: The message is still safe because it is encrypted with one of the strongest nowadays known algorithms, **AES-256-CBC**.

## Getting Started

### Prerequisites

*Eclipse* requires the excellent command line tool **ExifTool** by *Phil Harvey*.

```
sudo apt update
sudo apt install -y exiftool
```
Other packages will be automatically installed through the following instructions.

### Installing

First download the repository, and run:

```
python3 setup.py install
```

## Running Eclipse
You can use *Eclipse* in three different ways:
### As **Interactive Command-Line tool**:
```
python3 -m eclipse --interactive
```
![Screen1](eclipse/resources/screen1.png)

![Screen2](eclipse/resources/screen2.png)


### As **standard Command-Line tool**:
#### Hiding mode
```
python3 -m eclipse hide [--stealthy] --image <image-path> --message <message-txt> --code <seed> --output <path>
```
Example:
```
python3 -m eclipse hide -i "eclipse/resources/test_image.jpg" -m "SECRET MESSAGE" -c 20 -o "eclipse/resources/stego_image.png"
```
You will be asked to prompt a password (in this example "password" was used).
#### Message extracting mode
```
python3 -m eclipse eclipse extract [--stealthy | --output <path>] --image <image-path> --code <seed>
```
Example:
```
python3 -m eclipse extract -i "eclipse/resources/stego_image.png" -c 20
```
You will be asked to prompt the password used before.

### As **module**:
```
from eclipse.src.backend import encrypt_message, decrypt_message
```
Refers to **documentation** for the usage of single functions and methods.

### Help:
For help, type:
```
python3 -m eclipse -h
```
![Screen3](eclipse/resources/screen3.png)


## Running the tests

Tests are now under development, if you want to contribute, please read the section **Contributing**.


## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on the code of conduct, and the process for submitting pull requests to us.


## Authors

* **Mark Diamantino Caribé** - BA3 Computer Science - Université Libre de Bruxelles
	* mdcaribe@protonmail.com
	* mark.diamantino.caribe@ulb.ac.be

See also the list of [contributors](https://github.com/mdiamantino/eclipse/contributors) who participated in this project.

## License

This project is licensed under the *GNU AFFERO GENERAL PUBLIC LICENSE* - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgments
