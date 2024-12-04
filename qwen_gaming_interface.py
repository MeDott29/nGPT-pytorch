import ollama
from colorama import init, Fore, Style
from typing import List, Dict
import json
import textwrap
import logging

class DataChatbot:
    cache = {}

    def __init__(self, model: str = "llama3.2:1b"):
        init()  # Initialize colorama
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.client = ollama.Client()
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self.game_data = self.load_game_data()

    def chunk_json(self, json_obj, chunk_size=100):
        items = json.dumps(json_obj, indent=2).split('\n')
        chunks = []
        current_chunk = ""
        for item in items:
            if len(current_chunk) + len(item) + 1 > chunk_size:
                chunks.append(current_chunk)
                current_chunk = item
            else:
                if current_chunk:
                    current_chunk += '\n' + item
                else:
                    current_chunk = item
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def load_game_data(self) -> Dict:
        if 'game_data' in self.cache:
            return self.cache['game_data']

        data = {
            'games': [],
            'players': [],
            'chunked_games': [],
            'chunked_players': []
        }

        # Load game history
        try:
            with open('data/game_history.jsonl', 'r') as f:
                for line in f:
                    game_data = json.loads(line)
                    data['games'].append(game_data)
                    chunks = self.chunk_json(game_data)
                    data['chunked_games'].append(chunks)
        except Exception as e:
            self.logger.error(f"Error loading game history: {e}")

        # Load player data
        try:
            with open('data/players.jsonl', 'r') as f:
                for line in f:
                    player_data = json.loads(line)
                    data['players'].append(player_data)
                    chunks = self.chunk_json(player_data)
                    data['chunked_players'].append(chunks)
        except Exception as e:
            self.logger.error(f"Error loading player data: {e}")

        self.cache['game_data'] = data
        return data

    def display_chunked_data(self, data_type: str = 'games', index: int = 0):
        chunks = self.game_data[f'chunked_{data_type}'][index]
        print(f"\n{Fore.YELLOW}Displaying {data_type} record {index + 1} in chunks:{Style.RESET_ALL}")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n{Fore.CYAN}Chunk {i}:{Style.RESET_ALL}")
            print(chunk)

    def summarize_data(self, data_type: str, index: int):
        if data_type == 'games':
            game = self.game_data['games'][index]
            summary = (
                f"Game ID: {game.get('id', 'N/A')}, "
                f"Players: {len(game.get('players', []))}, "
                f"Pot Size: {game.get('pot_size', 'N/A')}"
            )
        elif data_type == 'players':
            player = self.game_data['players'][index]
            summary = (
                f"Player Name: {player.get('name', 'N/A')}, "
                f"Wins: {player.get('wins', 'N/A')}, "
                f"Losses: {player.get('losses', 'N/A')}"
            )
        else:
            summary = "Invalid data type"
        return summary

    def generate_response(self, user_input: str) -> str:
        # Create context from conversation history and data summary
        context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history[-5:]
        ])

        # Add data context with first chunk of first record for each type
        data_context = f"""
        Available game data:
        - {len(self.game_data['games'])} games in history
        - {len(self.game_data['players'])} player records

        Sample Game Data (first chunk):
        {self.game_data['chunked_games'][0][0] if self.game_data['chunked_games'] else '{}'}

        Sample Player Data (first chunk):
        {self.game_data['chunked_players'][0][0] if self.game_data['chunked_players'] else '{}'}
        """

        prompt = f"""You are an AI assistant analyzing poker game data in chunks. 
        Generate a focused response that:
        - Directly answers the user's question about the game data
        - Uses clear and concise language
        - References specific data points when relevant
        - Provides statistical insights when appropriate

        Available Data Context:
        {data_context}

        Conversation Context:
        {context}

        User Question: {user_input}
        """

        response = self.client.generate(self.model, prompt=prompt)
        return response['response']

    def chat(self):
        menu = """
        Welcome to the Game Data Chatbot!
        1. Show game [number]
        2. Show player [number]
        3. Summarize game [number]
        4. Summarize player [number]
        5. Ask a question about the game data
        6. Exit
        """
        print(menu)
        while True:
            user_input = input("Enter your choice: ")
            if '6' in user_input.lower() or 'exit' in user_input.lower():
                break
            elif user_input.startswith('1'):
                try:
                    index = int(user_input.split()[-1]) - 1
                    if 0 <= index < len(self.game_data['games']):
                        self.display_chunked_data('games', index)
                    else:
                        print(f"{Fore.RED}Invalid game number. Please choose between 1 and {len(self.game_data['games'])}{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please provide a valid game number{Style.RESET_ALL}")
            elif user_input.startswith('2'):
                try:
                    index = int(user_input.split()[-1]) - 1
                    if 0 <= index < len(self.game_data['players']):
                        self.display_chunked_data('players', index)
                    else:
                        print(f"{Fore.RED}Invalid player number. Please choose between 1 and {len(self.game_data['players'])}{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please provide a valid player number{Style.RESET_ALL}")
            elif user_input.startswith('3'):
                try:
                    index = int(user_input.split()[-1]) - 1
                    if 0 <= index < len(self.game_data['games']):
                        summary = self.summarize_data('games', index)
                        print(f"{Fore.BLUE}Summary: {summary}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Invalid game number. Please choose between 1 and {len(self.game_data['games'])}{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please provide a valid game number{Style.RESET_ALL}")
            elif user_input.startswith('4'):
                try:
                    index = int(user_input.split()[-1]) - 1
                    if 0 <= index < len(self.game_data['players']):
                        summary = self.summarize_data('players', index)
                        print(f"{Fore.BLUE}Summary: {summary}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Invalid player number. Please choose between 1 and {len(self.game_data['players'])}{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please provide a valid player number{Style.RESET_ALL}")
            elif user_input.startswith('5'):
                user_question = input(f"{Fore.GREEN}Ask your question: {Style.RESET_ALL}")
                self.conversation_history.append({"role": "user", "content": user_question})
                try:
                    response = self.generate_response(user_question)
                    print(f"{Fore.BLUE}Chatbot: {response}{Style.RESET_ALL}")
                    self.conversation_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    self.logger.error(f"Error generating response: {e}")
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    chatbot = DataChatbot()
    chatbot.chat()