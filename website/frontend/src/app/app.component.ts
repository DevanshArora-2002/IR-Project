import { Component } from '@angular/core';
import { ChatService } from './chat.service';

interface Message {
  text: string;
  isUser: boolean;
  profileImage: string;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent {
  title = 'frontend';
  newMessage = '';
  messages: Message[] = [];
  text1: string = 'What is Service Charge?';
  text2: string =
    'Sure! I can help you with that. Please provide me with your email address.';
  text3: string =
    'Thank you! I have sent you an email with the requested information.';
  text4: string =
    'I am sorry, but I am unable to help you with that. Please try again later.';

  constructor(private chatService: ChatService) {}

  ngOnInit() {
    this.messages = [];
    this.newMessage = '';
  }

  sendMessage() {
    if (this.newMessage.trim() !== '') {
      // Add the user's message to the chat window
      this.messages.push({
        text: this.newMessage,
        isUser: true,
        profileImage: 'user.png',
      });

      // Send the user's message to the backend using the service
      this.chatService.sendMessage(this.newMessage).subscribe((response) => {
        // Add the response from the backend to the chat window
        this.messages.push({
          text: response.response,
          isUser: false,
          profileImage: 'inlavs.png',
        });
      });

      // Clear the input field after sending the message
      this.newMessage = '';
    }
  }

  sendMessage2(message: string) {
    if (message !== '') {
      // Add the user's message to the chat window
      this.messages.push({
        text: message,
        isUser: true,
        profileImage: 'user.png',
      });

      // Send the user's message to the backend using the service
      this.chatService.sendMessage(message).subscribe((response) => {
        // Add the response from the backend to the chat window
        this.messages.push({
          text: response.response,
          isUser: false,
          profileImage: 'inlavs.png',
        });
      });

      // Clear the input field after sending the message
      this.newMessage = '';
    }
  }

  newChat() {
    this.ngOnInit();
  }
}
