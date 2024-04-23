import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private chatbotUrl = 'http://localhost:8000/chatbot'; // URL of your backend chatbot endpoint

  constructor(private http: HttpClient) { }

  // Method to send message to backend and receive response
  sendMessage(message: string): Observable<any> {
    return this.http.post<any>(this.chatbotUrl, { message });
  }
}
