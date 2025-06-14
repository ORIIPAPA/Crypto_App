import streamlit as st
from streamlit_option_menu import option_menu
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import yfinance as yf
import time
from bs4 import BeautifulSoup
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure NLTK
nltk.download(['punkt', 'stopwords', 'vader_lexicon'])
sia = SentimentIntensityAnalyzer()

# Initialize Hugging Face sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis")

# API Keys
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "your_api_key_here")
CRYPTO_PANIC_API_KEY = os.getenv("CRYPTO_PANIC_API_KEY", "your_api_key_here")

# Constants
SUPPORTED_COINS = ['bitcoin', 'ethereum', 'cardano', 'solana', 'polkadot', 
                  'ripple', 'dogecoin', 'avalanche', 'polygon', 'cosmos']

# Configure Streamlit
st.set_page_config(
    page_title="CryptoChief Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache expensive functions
@st.cache_data(ttl=3600)
def get_sustainability_data():
    """Get sustainability data from multiple sources"""
    try:
        # First try Cambridge Bitcoin Electricity Consumption Index
        cambridge_data = requests.get("https://ccaf.io/cbeci/api/v1/index").json()
        btc_energy = cambridge_data.get('current', {}).get('consumption', 0)
        
        # Then try Crypto Sustainability Index (mock - in reality would use API)
        sustainability_scores = {
            "bitcoin": {"energy_use": btc_energy, "sustainability_score": 3, "carbon_footprint": 500},
            "ethereum": {"energy_use": 0.0026 * btc_energy, "sustainability_score": 7, "carbon_footprint": 50},
            "cardano": {"energy_use": 0.0001 * btc_energy, "sustainability_score": 9, "carbon_footprint": 5},
            "solana": {"energy_use": 0.0005 * btc_energy, "sustainability_score": 8, "carbon_footprint": 10},
            "polkadot": {"energy_use": 0.0003 * btc_energy, "sustainability_score": 8, "carbon_footprint": 8}
        }
        return sustainability_scores
    except:
        # Fallback data
        return {
            "bitcoin": {"energy_use": "high", "sustainability_score": 3},
            "ethereum": {"energy_use": "medium", "sustainability_score": 6},
            "cardano": {"energy_use": "low", "sustainability_score": 8},
            "solana": {"energy_use": "low", "sustainability_score": 7},
            "polkadot": {"energy_use": "low", "sustainability_score": 8},
        }

@st.cache_data(ttl=300)
def get_crypto_data(coin_name):
    """Get comprehensive crypto data from CoinGecko"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_name.lower()}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false"
        headers = {"x-cg-demo-api-key": COINGECKO_API_KEY}
        response = requests.get(url, headers=headers).json()
        
        market_data = response.get('market_data', {})
        return {
            "price": market_data.get('current_price', {}).get('usd', 0),
            "price_change_24h": market_data.get('price_change_percentage_24h', 0),
            "market_cap": market_data.get('market_cap', {}).get('usd', 0),
            "volume": market_data.get('total_volume', {}).get('usd', 0),
            "ath": market_data.get('ath', {}).get('usd', 0),
            "atl": market_data.get('atl', {}).get('usd', 0),
            "circulating_supply": market_data.get('circulating_supply', 0),
            "image": response.get('image', {}).get('large', '')
        }
    except Exception as e:
        st.error(f"Error fetching data for {coin_name}: {str(e)}")
        return None

@st.cache_data(ttl=600)
def get_crypto_news():
    """Get latest crypto news from CryptoPanic"""
    try:
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTO_PANIC_API_KEY}&filter=hot"
        response = requests.get(url).json()
        return response.get('results', [])[:5]
    except:
        return []

@st.cache_data(ttl=3600)
def get_historical_data(coin_name, days=30):
    """Get historical price data"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_name}/market_chart?vs_currency=usd&days={days}"
        headers = {"x-cg-demo-api-key": COINGECKO_API_KEY}
        response = requests.get(url, headers=headers).json()
        prices = response.get('prices', [])
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except:
        return pd.DataFrame()

def analyze_sentiment(text):
    """Perform sentiment analysis using both NLTK and Hugging Face"""
    # NLTK VADER
    vader_score = sia.polarity_scores(text)
    
    # Hugging Face
    try:
        hf_result = sentiment_pipeline(text)[0]
        hf_score = {'label': hf_result['label'], 'score': hf_result['score']}
    except:
        hf_score = {'label': 'NEUTRAL', 'score': 0.5}
    
    return {
        'vader': vader_score,
        'huggingface': hf_score,
        'composite': (vader_score['compound'] + (hf_score['score'] if hf_score['label'] == 'POSITIVE' else -hf_score['score'])) / 2
    }

def extract_keywords(user_input):
    """Enhanced keyword extraction with entity recognition"""
    tokens = nltk.word_tokenize(user_input.lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))
    keywords = [word for word in tokens if word not in stop_words and word.isalpha()]
    
    # Simple entity recognition
    entities = {
        'coins': [word for word in keywords if word in SUPPORTED_COINS],
        'actions': [word for word in keywords if word in ['buy', 'sell', 'hold', 'invest', 'trade']],
        'metrics': [word for word in keywords if word in ['price', 'volume', 'market', 'cap', 'supply']],
        'time': [word for word in keywords if word in ['today', 'week', 'month', 'year']]
    }
    
    return {'keywords': keywords, 'entities': entities}

def generate_response(user_query, portfolio=None):
    """Enhanced response generation with context awareness"""
    analysis = extract_keywords(user_query)
    sentiment = analyze_sentiment(user_query)
    
    # Check for greetings
    if any(word in analysis['keywords'] for word in ['hi', 'hello', 'hey']):
        return {"type": "text", "content": "Hello! I'm CryptoChief. How can I help you with crypto today?"}
    
    # Check for thank you
    elif any(word in analysis['keywords'] for word in ['thanks', 'thank']):
        return {"type": "text", "content": "You're welcome! Is there anything else you'd like to know?"}
    
    # Sustainability queries
    elif any(word in analysis['keywords'] for word in ['sustainable', 'eco', 'green', 'environment']):
        sustainability_db = get_sustainability_data()
        best = max(sustainability_db.items(), key=lambda x: x[1]["sustainability_score"])
        worst = min(sustainability_db.items(), key=lambda x: x[1]["sustainability_score"])
        
        response = f"""
        **Sustainability Report** 🌱
        
        - 🏆 **Best**: {best[0].title()} (Score: {best[1]['sustainability_score']}/10)
        - 🚫 **Worst**: {worst[0].title()} (Score: {worst[1]['sustainability_score']}/10)
        
        *Tip: Consider {best[0].title()} for eco-conscious investing!*
        """
        return {"type": "markdown", "content": response}
    
    # Price queries
    elif any(word in analysis['keywords'] for word in ['price', 'value', 'worth']):
        if analysis['entities']['coins']:
            coin = analysis['entities']['coins'][0]
            data = get_crypto_data(coin)
            if data:
                hist_data = get_historical_data(coin)
                
                if not hist_data.empty:
                    fig = px.line(hist_data, x='date', y='price', 
                                 title=f'{coin.title()} Price Last 30 Days')
                    
                    response = f"""
                    **{coin.title()} Price Data** 💰
                    
                    - Current Price: ${data['price']:,.2f}
                    - 24h Change: {data['price_change_24h']:.2f}%
                    - Market Cap: ${data['market_cap']:,.0f}
                    - 24h Volume: ${data['volume']:,.0f}
                    
                    *Chart shows 30-day trend*
                    """
                    return {"type": "mixed", "text": response, "plot": fig}
        
        # If no specific coin mentioned, show top coins
        top_coins = []
        for coin in SUPPORTED_COINS[:5]:
            data = get_crypto_data(coin)
            if data:
                top_coins.append(f"{coin.title()}: ${data['price']:,.2f} ({data['price_change_24h']:+.2f}%)")
        
        return {"type": "markdown", "content": "**Current Top Crypto Prices**\n\n" + "\n".join(top_coins)}
    
    # Portfolio queries
    elif portfolio and any(word in analysis['keywords'] for word in ['portfolio', 'holdings', 'investment']):
        if not portfolio.empty:
            portfolio['value'] = portfolio['amount'] * portfolio['current_price']
            total_value = portfolio['value'].sum()
            
            fig = px.pie(portfolio, names='coin', values='value', 
                         title='Portfolio Allocation')
            
            response = f"""
            **Your Portfolio Summary** 📊
            
            - Total Value: ${total_value:,.2f}
            - Top Holding: {portfolio.loc[portfolio['value'].idxmax()]['coin']}
            
            *Allocation chart below*
            """
            return {"type": "mixed", "text": response, "plot": fig}
        else:
            return {"type": "text", "content": "Your portfolio is currently empty. Would you like to add some holdings?"}
    
    # News queries
    elif any(word in analysis['keywords'] for word in ['news', 'update', 'happening']):
        news_items = get_crypto_news()
        if news_items:
            response = "**Latest Crypto News** 📰\n\n"
            for item in news_items:
                response += f"- [{item['title']}]({item['url']})\n"
            return {"type": "markdown", "content": response}
        else:
            return {"type": "text", "content": "Couldn't fetch the latest news right now. Please try again later."}
    
    # Sentiment analysis
    elif any(word in analysis['keywords'] for word in ['sentiment', 'mood', 'feeling']):
        response = f"""
        **Sentiment Analysis** 😊😐😠
        
        - VADER: {sentiment['vader']['compound']:.2f} ({'positive' if sentiment['vader']['compound'] > 0.05 else 'negative' if sentiment['vader']['compound'] < -0.05 else 'neutral'})
        - HuggingFace: {sentiment['huggingface']['label']} ({sentiment['huggingface']['score']:.2f})
        - Composite: {'Positive' if sentiment['composite'] > 0 else 'Negative' if sentiment['composite'] < 0 else 'Neutral'}
        
        *Based on your query: "{user_query}"*
        """
        return {"type": "markdown", "content": response}
    
    # Default response
    else:
        return {"type": "text", "content": "I'm not sure I understand. Try asking about prices, sustainability, news, or your portfolio."}

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['coin', 'amount', 'purchase_price', 'current_price'])

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# UI Components
def show_home():
    st.title("🚀 CryptoChief Pro")
    st.markdown("""
    Your intelligent cryptocurrency assistant providing:
    - Real-time market data 📈
    - Sustainability insights 🌱
    - Portfolio tracking 💼
    - Sentiment analysis 😊😐😠
    - Latest crypto news 📰
    """)
    
    # Market overview
    st.subheader("Market Overview")
    cols = st.columns(5)
    for i, coin in enumerate(SUPPORTED_COINS[:5]):
        with cols[i]:
            data = get_crypto_data(coin)
            if data:
                st.image(data['image'], width=50)
                st.metric(
                    label=coin.title(),
                    value=f"${data['price']:,.2f}",
                    delta=f"{data['price_change_24h']:.2f}%"
                )

def show_chat():
    st.title("💬 Chat with CryptoChief")
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.chat_message("user").write(msg['content'])
            else:
                with st.chat_message("assistant"):
                    if msg['response']['type'] == 'text':
                        st.write(msg['response']['content'])
                    elif msg['response']['type'] == 'markdown':
                        st.markdown(msg['response']['content'])
                    elif msg['response']['type'] == 'mixed':
                        st.markdown(msg['response']['text'])
                        st.plotly_chart(msg['response']['plot'], use_container_width=True)
    
    # User input
    if prompt := st.chat_input("Ask me about crypto..."):
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': prompt,
            'time': datetime.now()
        })
        
        # Generate response
        with st.spinner("Analyzing..."):
            response = generate_response(prompt, st.session_state.portfolio)
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': prompt,
                'response': response,
                'time': datetime.now()
            })
            
            # Rerun to update display
            st.rerun()

def show_portfolio():
    st.title("📊 Portfolio Tracker")
    
    # Add to portfolio
    with st.expander("Add to Portfolio"):
        col1, col2, col3 = st.columns(3)
        with col1:
            coin = st.selectbox("Coin", SUPPORTED_COINS)
        with col2:
            amount = st.number_input("Amount", min_value=0.0, step=0.01)
        with col3:
            purchase_price = st.number_input("Purchase Price (USD)", min_value=0.0, step=0.01)
        
        if st.button("Add to Portfolio"):
            current_data = get_crypto_data(coin)
            if current_data:
                new_row = {
                    'coin': coin,
                    'amount': amount,
                    'purchase_price': purchase_price,
                    'current_price': current_data['price']
                }
                
                if st.session_state.portfolio.empty:
                    st.session_state.portfolio = pd.DataFrame([new_row])
                else:
                    st.session_state.portfolio = pd.concat([
                        st.session_state.portfolio,
                        pd.DataFrame([new_row])
                    ], ignore_index=True)
                st.success("Added to portfolio!")
    
    # Display portfolio
    if not st.session_state.portfolio.empty:
        # Update current prices
        for i, row in st.session_state.portfolio.iterrows():
            data = get_crypto_data(row['coin'])
            if data:
                st.session_state.portfolio.at[i, 'current_price'] = data['price']
        
        # Calculate metrics
        st.session_state.portfolio['value'] = st.session_state.portfolio['amount'] * st.session_state.portfolio['current_price']
        st.session_state.portfolio['cost'] = st.session_state.portfolio['amount'] * st.session_state.portfolio['purchase_price']
        st.session_state.portfolio['pnl'] = st.session_state.portfolio['value'] - st.session_state.portfolio['cost']
        st.session_state.portfolio['pnl_pct'] = (st.session_state.portfolio['pnl'] / st.session_state.portfolio['cost']) * 100
        
        # Display
        st.dataframe(
            st.session_state.portfolio.style
            .format({
                'current_price': '${:,.2f}',
                'purchase_price': '${:,.2f}',
                'value': '${:,.2f}',
                'cost': '${:,.2f}',
                'pnl': '${:,.2f}',
                'pnl_pct': '{:.2f}%'
            }),
            use_container_width=True
        )
        
        # Portfolio metrics
        total_value = st.session_state.portfolio['value'].sum()
        total_cost = st.session_state.portfolio['cost'].sum()
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Value", f"${total_value:,.2f}")
        col2.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl_pct:.2f}%")
        
        # Allocation chart
        st.plotly_chart(
            px.pie(st.session_state.portfolio, 
                  names='coin', 
                  values='value',
                  title='Portfolio Allocation'),
            use_container_width=True
        )
    else:
        st.info("Your portfolio is empty. Add some holdings to get started!")

def show_analysis():
    st.title("🔍 Market Analysis")
    
    # Coin selector
    coin = st.selectbox("Select Coin", SUPPORTED_COINS)
    
    if coin:
        data = get_crypto_data(coin)
        hist_data = get_historical_data(coin, days=90)
        
        if data and not hist_data.empty:
            # Price chart
            st.subheader(f"{coin.title()} Price Trend")
            fig = px.line(hist_data, x='date', y='price', 
                         title=f'{coin.title()} Price Last 90 Days')
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${data['price']:,.2f}")
            col2.metric("24h Change", f"{data['price_change_24h']:.2f}%")
            col3.metric("Market Cap", f"${data['market_cap']:,.0f}")
            
            # Sustainability
            sustainability_db = get_sustainability_data()
            if coin in sustainability_db:
                st.subheader("Sustainability Report")
                cols = st.columns(2)
                with cols[0]:
                    st.metric("Sustainability Score", 
                              f"{sustainability_db[coin]['sustainability_score']}/10")
                with cols[1]:
                    st.metric("Estimated Annual Energy Use", 
                              f"{sustainability_db[coin]['energy_use']:,.0f} kWh")
                
                # Comparison
                st.plotly_chart(
                    px.bar(
                        x=list(sustainability_db.keys()),
                        y=[v['sustainability_score'] for v in sustainability_db.values()],
                        title="Sustainability Comparison"
                    ),
                    use_container_width=True
                )

# Sidebar navigation
with st.sidebar:
    st.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=100)
    st.title("Navigation")
    selected = option_menu(
        menu_title=None,
        options=["Home", "Chat", "Portfolio", "Analysis"],
        icons=["house", "chat", "wallet", "graph-up"],
        default_index=0
    )
    
    st.markdown("---")
    st.markdown("""
    **About CryptoChief**  
    An intelligent crypto assistant  
    using real-time data and AI  
    to help you make better decisions.
    """)
    
    st.markdown("---")
    st.markdown("**Disclaimer**")
    st.caption("Cryptocurrency investments are volatile and risky. This tool provides information, not financial advice.")

# Page routing
if selected == "Home":
    show_home()
elif selected == "Chat":
    show_chat()
elif selected == "Portfolio":
    show_portfolio()
elif selected == "Analysis":
    show_analysis()
