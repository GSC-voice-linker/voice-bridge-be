
# Voice Bridge BE

## Overview
Voice Bridge is a transformative communication platform designed to facilitate seamless interaction for the deaf and hard-of-hearing community. This project leverages advanced technology to provide bidirectional communication translation, enabling users to convert speech to text and sign language videos to natural language text.

## Technical Stack
The backend API server for Voice Bridge is developed using Django, a high-level Python web framework that encourages rapid development and clean, pragmatic design. The project is structured around the main Django project named `bridge`, containing two significant applications: `stt` and `sign_translation`.

### Applications
- **stt (Speech-to-Text)**: This application is focused on converting voice inputs into text. By processing spoken language, it provides a textual representation of the speech, making it accessible for users who rely on text for communication.

- **sign_translation**: This application takes sign language videos as input and utilizes Machine Learning (ML) algorithms to translate the signs. Further processing through Large Language Models (LLMs) refines the translation into natural, coherent sentences, presented in text format for easy comprehension.

## Getting Started

### Prerequisites
- Python 3.11
- Django 5.0.1
- google-api-python-client 2.115.0
- google-cloud-speech 2.23.0

### Running the Server
To start the Django development server, run the following command inside the project directory:
```bash
python manage.py runserver
```
This will start the server on the default port (usually 8000), accessible via `http://localhost:8000`.

## Usage
- **For Speech-to-Text**: Send a POST request to `/stt/` with the voice input to receive the textual translation.
- **For Sign Language Translation**: Send a POST request to `/sign_translation/` with the sign language video to get the translated text.
