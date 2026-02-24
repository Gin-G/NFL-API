import { apiClient } from './client'
import type { ChatMessage, ChatResponse } from './types'

export type { ChatMessage, ChatResponse }

export async function sendChatMessage(
  message: string,
  history: ChatMessage[],
): Promise<ChatResponse> {
  const { data } = await apiClient.post<ChatResponse>('/chat/', { message, history })
  return data
}
