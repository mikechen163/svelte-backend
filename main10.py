from fastapi import FastAPI, HTTPException  
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel  
from typing import Optional, Dict, Any  
import numpy as np  
import json  
from decimal import Decimal  
import matplotlib  
matplotlib.use('Agg')  
from openbb import obb  
import pandas as pd  
import mplfinance as mpf  
from datetime import datetime, timedelta  
import base64  
import io  

# Initialize FastAPI app  
app = FastAPI()  

# Configure CORS  
app.add_middleware(  
    CORSMiddleware,  
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://192.168.191.56:5173"],  
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"],  
)  

class CustomJSONEncoder(json.JSONEncoder):  
    def default(self, obj):  
        if isinstance(obj, (np.integer, np.floating)):  
            return float(obj) if not np.isnan(obj) and not np.isinf(obj) else None  
        if isinstance(obj, datetime):  
            return obj.isoformat()  
        if isinstance(obj, Decimal):  
            return str(obj)  
        if pd.isna(obj):  
            return None  
        return super().default(obj)  



# Add health check endpoint  
@app.get("/health")  
async def health_check():  
    return {"status": "healthy"}  


def clean_dataframe(df):  
    """Clean DataFrame to ensure JSON compatibility"""  
    try:  
        # Make a copy to avoid modifying the original dataframe  
        df = df.copy()  

        # Replace inf/-inf with NaN first  
        df = df.replace([np.inf, -np.inf], np.nan)  

        # Convert numpy types to Python native types and handle NaN values  
        for column in df.columns:  
            if df[column].dtype in [np.float64, np.float32]:  
                df[column] = df[column].apply(  
                    lambda x: float(x) if pd.notnull(x) and not np.isinf(x) else None  
                )  
            elif df[column].dtype in [np.int64, np.int32]:  
                df[column] = df[column].apply(  
                    lambda x: int(x) if pd.notnull(x) else None  
                )  
            else:  
                # For other types, just replace NaN with None  
                df[column] = df[column].where(pd.notnull(df[column]), None)  

        return df  
    except Exception as e:  
        print(f"Error in clean_dataframe: {str(e)}")  
        return df  






  
def fetch_stock_data(ticker):  
    """Fetch stock price data"""  
    end_date = datetime.now().date()  
    start_date = (datetime.now() - timedelta(days=365)).date()  

    try:  
        stock_data = obb.equity.price.historical(  
            ticker,  
            start_date=start_date.strftime("%Y-%m-%d"),  
            end_date=end_date.strftime("%Y-%m-%d")  
        )  

        # Convert to DataFrame  
        df = pd.DataFrame(stock_data.to_dict())  

        if df.empty:  
            return None, "No data available for this ticker"  

        # Ensure the index is a DatetimeIndex  
        if 'date' in df.columns:  
            df['date'] = pd.to_datetime(df['date'])  
            df.set_index('date', inplace=True)  
        else:  
            return None, "Missing date column in stock data"  

        # Verify required columns exist  
        required_columns = ['open', 'high', 'low', 'close', 'volume']  
        if not all(col in df.columns for col in required_columns):  
            return None, f"Missing required columns. Available columns: {df.columns}"  

        return df, None  
    except Exception as e:  
        return None, str(e)  


def fetch_financial_data(ticker):  
    """Fetch financial statement data"""  
    try:  
        # Try fetching income statement without specifying frequency  
        income_statement = obb.equity.fundamental.income(ticker)  
        results = income_statement.results  

        if not results or len(results) == 0:  
            return None, "No financial data available for this ticker"  

        data = [item.model_dump() for item in results]  
        df = pd.DataFrame(data)  

        if 'period_ending' in df.columns:  
            df['period_ending'] = pd.to_datetime(df['period_ending'])  
            df.set_index('period_ending', inplace=True)  
        else:  
            return None, "No 'period_ending' column in financial data"  

        df = df.sort_index(ascending=False)  
        df = df.head(12)  

        metrics_to_include = [  
            'total_revenue',  
            'cost_of_revenue',  
            'gross_profit',  
            'operating_income',  
            'net_income',  
            'ebitda',  
            'normalized_ebitda',  
        ]  

        # Filter and clean the data  
        available_metrics = [m for m in metrics_to_include if m in df.columns]  
        if available_metrics:  
            df = df[available_metrics]  
            # Convert all numeric columns to float and handle inf/nan  
            for col in df.columns:  
                df[col] = pd.to_numeric(df[col], errors='coerce')  
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)  
                df[col] = df[col].apply(lambda x: float(x) if pd.notnull(x) else None)  
            return df, None  

        return None, "No relevant financial metrics available for this ticker"  

    except Exception as e:  
        print(f"Debug - Error details: {str(e)}")  
        return None, f"Error fetching financial data: {str(e)}"  

def create_candlestick_chart(df, ticker):  
    """Create candlestick chart"""  
    try:  
        # Ensure required columns exist and contain valid data  
        required_columns = ['open', 'high', 'low', 'close', 'volume']  
        if not all(col in df.columns for col in required_columns):  
            print(f"Missing required columns. Available columns: {df.columns}")  
            return None  

        # Clean data specifically for plotting  
        df_plot = df.copy()  

        # Handle NaN values using forward fill and backward fill  
        df_plot = df_plot.ffill().bfill()  

        # Convert all numeric columns to float and handle inf/nan  
        for col in required_columns:  
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')  
            df_plot[col] = df_plot[col].replace([np.inf, -np.inf], np.nan)  

        # Drop any remaining problematic rows  
        df_plot = df_plot.dropna(subset=required_columns)  

        if df_plot.empty:  
            print("No valid data remaining after cleaning for plot")  
            return None  

        buf = io.BytesIO()  

        mc = mpf.make_marketcolors(  
            up='green', down='red',  
            edge='inherit', wick='inherit',  
            volume='in', ohlc='inherit'  
        )  

        s = mpf.make_mpf_style(  
            marketcolors=mc,  
            gridstyle='dotted',  
            y_on_right=False  
        )  

        fig, axlist = mpf.plot(  
            df_plot,  
            type='candle',  
            title=f'\n{ticker} Stock Price',  
            volume=True,  
            style=s,  
            figsize=(15, 10),  
            returnfig=True  
        )  

        fig.savefig(buf, format='png', bbox_inches='tight')  
        buf.seek(0)  
        chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')  
        buf.close()  
        return chart_base64  
    except Exception as e:  
        print(f"Error creating chart: {str(e)}")  
        return None  

@app.get("/api/stock/{ticker}")  
async def get_stock_data(ticker: str):  
    ticker = ticker.upper()  

    try:  
        # Fetch stock price data  
        df, error = fetch_stock_data(ticker)  
        if error:  
            raise HTTPException(status_code=400, detail=error)  

        # Fetch financial data  
        financials, fin_error = fetch_financial_data(ticker)  

        response = {"error": None}  

        if df is not None and not df.empty:  
            # Generate chart  
            chart_image = create_candlestick_chart(df, ticker)  
            if chart_image:  
                response["chart_image"] = chart_image  

        if financials is not None and not financials.empty:  
            try:  
                # Convert datetime index to string format  
                financials.index = financials.index.strftime('%Y-%m-%d')  
                # Convert to records and handle any remaining non-JSON-serializable values  
                financials_dict = []  
                for record in financials.reset_index().to_dict('records'):  
                    cleaned_record = {}  
                    for k, v in record.items():  
                        if isinstance(v, (float, np.float64)):  
                            if np.isnan(v) or np.isinf(v):  
                                cleaned_record[k] = None  
                            else:  
                                cleaned_record[k] = float(v)  
                        else:  
                            cleaned_record[k] = v  
                    financials_dict.append(cleaned_record)  
                response["financials"] = financials_dict  
            except Exception as e:  
                print(f"Error converting financials to dict: {str(e)}")  

        if not response.get("chart_image") and not response.get("financials"):  
            response["error"] = "No data available for this ticker"  

        return response  

    except Exception as e:  
        print(f"Error processing request: {str(e)}")  
        raise HTTPException(status_code=500, detail=str(e))          
  

  
