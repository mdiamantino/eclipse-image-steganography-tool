<img src="eclipse/resources/eclipse_icon.png" align="right" />

# Eclipse

Eclipse is a **steganography tool** that can be used to hide or extract a message from a given image.

## Goal
*Eclipse* aims to be an **efficient tool** for image steganography combining different concepts of computer science, such as **cryptography**, **compression** and **machine learning**.

## Detailed Principle
In contrast to secure communication, where an encrypted message is clearly exhchanged, the main goal of a steganography image is to let the exchange **go unnoticed**, so that it is difficult to detect the transfer of a message and even more challenging retrieving it.

In order to achieve a rasonable efficiency and securety level, *Eclipse* focuses on three foundamentals:
- **Avoiding image comparison** at  all costs: 
	- **Why:** If the *stego image* (the one containg the hidden message) can be compared to the *cover image* (the one without), differences could encourage inspections.
	- **How:**  *Eclipse* avoids, as far as possible, that the coverimage using two tools of **machine learning**:
		- **Image Augmentation**: The message is never hidden in the original image. Instead, *Eclipse* performs **random transformations** on it so that it could be really hard to generate the same image again and compare it to the stegoimage;
		- **Black-box adversarial attack** *[feature under developement]*: Image recognition tools could be used to search for the original image (and eventually proceed to bruteforce the above-mentioned point). *Eclipse* aims to black-box adversarial attacks, so most of image recognition systems would misidentify the possible stegoimage.
		- **Metadata suppression**:*Eclipse* deletes all EXIFS, in order to discourage image traceability such as **GPS coordinates**, **origin**, **author** and so forth.
- **Minimizing differences** between the two images: 
	- **Why:**  If *Eclipse* fails the first point and the original cover-image is found,  then it is crucial that differences are negligeable and imperceptibles, so that they could be easily associated to eventual transfer/compression operations.
	- **How:**  
		- *Eclipse* steganography technique is based on **discrete cosine transform**, which compared to the standard LSB technique, is **far more robust, safe and imperceptible**. In addition, the technique has been modified so that the message is hidden only **one bit/highest coefficient** -in order to reduce machine perceptibility- and exclusively in the **Cb is blue minus luma** component -which is the worst noticed by the humain eye.
		- The message is **randmly and uniformily distributed** in the image.
- **Message encryption**:
	- **Why**: When the first to points fail, the original message needs to be unreadable.
	- **How**: The message is still safe because it is encrypted with one of the strongest nowadays known algorithms, **AES-256-CBC**.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

