import streamlit as st
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

sustainability_db = {
    "bitcoin": {"energy_use": "high", "sustainability_score": 3},
    "ethereum": {"energy_use": "medium", "sustainability_score": 6},
    "cardano": {"energy_use": "low", "sustainability_score": 8},
    "world coin": {"energy_use": "low", "sustainability_score": 4},
}

def get_crypto_data(coin_name):
    try:
        response = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin_name.lower()}").json()
        price = response['market_data']['current_price']['usd']
        trend = response['market_data']['price_change_percentage_24h']
        market_cap_rank = response['market_cap_rank']
        return {
            "price": price,
            "trend": "rising" if trend > 0 else "falling",
            "market_cap_rank": market_cap_rank
        }
    except:
        return None

def extract_keywords(user_input):
    tokens = word_tokenize(user_input.lower())
    return [word for word in tokens if word not in stopwords.words('english')]

def respond_to_query(user_query):
    keywords = extract_keywords(user_query)
    if any(k in keywords for k in ['sustainable', 'eco', 'green']):
        best = max(sustainability_db, key=lambda x: sustainability_db[x]["sustainability_score"])
        return f"{best.title()} is a top pick for eco-conscious investing! ♻️"
    elif 'trending' in keywords or 'rising' in keywords:
        coins = ['bitcoin', 'ethereum', 'cardano', 'world coin']
        rising = [coin.title() for coin in coins if (data := get_crypto_data(coin)) and data['trend'] == 'rising']
        return f"📈 Currently rising: {', '.join(rising)}" if rising else "No strong upward trends right now."
    elif 'long-term' in keywords or 'growth' in keywords:
        coin = 'cardano'
        data = get_crypto_data(coin)
        if data and data['trend'] == 'rising' and sustainability_db[coin]['sustainability_score'] >= 7:
            return f"🚀 {coin.title()} is a sustainable, long-term growth opportunity."
    elif 'price' in keywords:
        result = []
        for coin in ['bitcoin', 'ethereum', 'cardano', 'world coin']:
            data = get_crypto_data(coin)
            if data:
                result.append(f"{coin.title()}: ${data['price']:.2f}")
        return "\n".join(result)
    return "🤔 Sorry, I couldn't understand that. Try asking about sustainability, price, or growth."

# UI
st.title("🤖 CryptoChief")
st.write("Ask about crypto prices, eco-friendliness, or long-term trends.")

query = st.text_input("You:")
if query:
    st.write("🧠 Thinking...")
    st.write(respond_to_query(query))

st.info("📌 Disclaimer: Crypto investments are risky. Always do your own research.")
