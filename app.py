import streamlit as st
import datetime
from langchain.chat_models import ChatOpenAI

# Streamlitの設定 (set_page_configは最初に呼び出す必要があります)
st.set_page_config(page_title="ダイエットサポートAI", layout="wide")
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
# Removed unused import: StreamlitCallbackHandler
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import LLMChain, ConversationChain
import pytz
import re

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAI APIキーの設定（環境変数から取得）
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI APIキーが設定されていません。環境変数 'OPENAI_API_KEY' を設定してください。アプリを終了します。")
    st.stop()  # Ensure the app stops gracefully if the API key is missing

# LangChainの設定
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    streaming=True,
    temperature=0.5
)

# メモリの設定
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# プロンプトテンプレートの設定
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

st.title("ダイエットサポートAI")

# サイドバーに過去の記録を表示
st.sidebar.header("📅 過去の記録")
if "records" not in st.session_state:
    st.session_state.records = {}

# タイムゾーンの設定
default_timezone = pytz.timezone("Asia/Tokyo")
selected_date = st.sidebar.date_input("日付を選択", datetime.datetime.now(default_timezone).date())
if selected_date in st.session_state.records:
    record = st.session_state.records[selected_date]
    st.sidebar.write(f"**朝食**: {record['breakfast']['content']} ({record['breakfast']['calories']} kcal)")
    st.sidebar.write(f"**昼食**: {record['lunch']['content']} ({record['lunch']['calories']} kcal)")
    st.sidebar.write(f"**夕食**: {record['dinner']['content']} ({record['dinner']['calories']} kcal)")
    st.sidebar.write(f"**その他**: {record['snacks']['content']} ({record['snacks']['calories']} kcal)")
    st.sidebar.write(f"**運動**: {record['exercise']['content']} ({record['exercise']['calories']} kcal)")
    st.sidebar.write(f"**摂取カロリー合計**: {record['total_intake']} kcal")
    st.sidebar.write(f"**カロリー収支**: {record['calorie_balance']} kcal")
else:
    st.sidebar.write("記録がありません。")

# ユーザープロファイルの入力
st.header("👤 ユーザープロファイル")

# セッションにユーザープロファイルが保存されていない場合のみ初期化
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False

# プロファイル編集モードの切り替え
if st.session_state.user_profile and not st.session_state.edit_mode:
    st.write("プロファイルが保存されています。以下の情報を使用します。")
    st.write(f"年齢: {st.session_state.user_profile['age']} 歳")
    st.write(f"性別: {st.session_state.user_profile['sex']}")
    st.write(f"身長: {st.session_state.user_profile['height']} cm")
    st.write(f"目標体重: {st.session_state.user_profile['target_weight']} kg")
    st.write(f"目標達成日: {st.session_state.user_profile['target_date']}")

    if st.button("プロファイルを編集"):
        st.session_state.edit_mode = True
else:
    st.write("プロファイルを入力または編集してください。")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("年齢", min_value=10, max_value=100, value=st.session_state.user_profile.get("age", 30))
        sex = st.selectbox("性別", ["男性", "女性"], index=["男性", "女性"].index(st.session_state.user_profile.get("sex", "男性")))
        height = st.number_input("身長 (cm)", min_value=100, max_value=250, value=st.session_state.user_profile.get("height", 170))
    with col2:
        target_weight = st.number_input("目標体重 (kg)", min_value=30, max_value=200, value=st.session_state.user_profile.get("target_weight", 60))
        target_date = st.date_input("目標達成日", st.session_state.user_profile.get("target_date", datetime.date.today() + datetime.timedelta(days=90)))

    if st.button("プロファイルを保存"):
        st.session_state.user_profile = {
            "age": age,
            "sex": sex,
            "height": height,
            "target_weight": target_weight,
            "target_date": target_date
        }
        st.session_state.edit_mode = False
        st.success("プロファイルを保存しました。")

# 現在の体重は毎回入力可能
current_weight = st.number_input("現在の体重 (kg)", min_value=30, max_value=200, value=65)

# 基礎代謝量（BMR）の計算
@st.cache_data
def calculate_bmr(sex, weight, height, age):
    if sex == "男性":
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

bmr = calculate_bmr(
    st.session_state.user_profile.get("sex", "男性"),
    current_weight,
    st.session_state.user_profile.get("height", 170),
    st.session_state.user_profile.get("age", 30)
)
st.write(f"基礎代謝量（BMR）は約 {bmr:.2f} kcal/日です。")

# LLMChainを使用してカロリーを計算する関数
def get_calories_from_chatgpt_with_llmchain(food_input):
    if not food_input.strip():
        return {"items": [], "total_calories": 0}  # 入力が空の場合は0を返す

    # ResponseSchemaを使用してスキーマを定義
    response_schemas = [
        ResponseSchema(name="items", description="食事のリストとそれぞれのカロリー"),
        ResponseSchema(name="total_calories", description="合計カロリー")
    ]

    # StructuredOutputParserの設定
    parser = StructuredOutputParser(response_schemas=response_schemas)

    # プロンプトテンプレートの設定
    template = (
        "ユーザーが入力した食事とそのカロリー、合計カロリーをJSON形式で出力してください。\n"
        "食事内容: {food_input}\n"
        "{format_instructions}"
    )
    prompt = PromptTemplate(
        input_variables=["food_input"],
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # LLMChainの設定
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # LLMChainを使用して応答を取得
    response = llm_chain.run({"food_input": food_input})
    st.write(f"ChatGPTの応答: {response}")  # 応答を確認したい場合に表示

    try:
        # パーサーを使用してJSON形式の応答を解析
        parsed_output = parser.parse(response)
        return parsed_output
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        return {"items": [], "total_calories": 0}  # 応答が不正な場合は0を返す

# カロリーを整数に変換する関数
def extract_calories(calorie_data):
    if isinstance(calorie_data, (int, float)):  # すでに数値の場合
        return int(calorie_data)
    elif isinstance(calorie_data, str):  # 文字列の場合
        match = re.search(r"\d+", calorie_data)  # 数字部分を抽出
        if match:
            return int(match.group())  # 数字部分を整数に変換
    return 0  # 数字が見つからない場合は0を返す

# 運動内容から消費カロリーを計算する関数（JSON形式対応）
def get_exercise_calories_from_chatgpt_with_llmchain(exercise_input):
    if not exercise_input.strip():
        return {"exercise_details": [], "total_calories": 0}  # 入力が空の場合は0を返す

    # ResponseSchemaを使用してスキーマを定義
    response_schemas = [
        ResponseSchema(name="exercise_details", description="運動のリストとそれぞれの消費カロリー"),
        ResponseSchema(name="total_calories", description="合計消費カロリー")
    ]

    # StructuredOutputParserの設定
    parser = StructuredOutputParser(response_schemas=response_schemas)

    # プロンプトテンプレートの設定
    template = (
        "ユーザーが入力した運動内容とその消費カロリー、合計消費カロリーをJSON形式で出力してください。\n"
        "運動内容: {exercise_input}\n"
        "{format_instructions}"
    )
    prompt = PromptTemplate(
        input_variables=["exercise_input"],
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # LLMChainの設定
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # LLMChainを使用して応答を取得
    response = llm_chain.run({"exercise_input": exercise_input})
    st.write(f"ChatGPTの応答: {response}")  # 応答を確認したい場合に表示

    try:
        # パーサーを使用してJSON形式の応答を解析
        parsed_output = parser.parse(response)
        return parsed_output
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        return {"exercise_details": [], "total_calories": 0}  # 応答が不正な場合は0を返す

# 本日の記録
st.header("🍴 本日の記録")

# 記録する日を選択するカレンダー機能
record_date = st.date_input("記録する日を選択してください", value=datetime.datetime.now(default_timezone).date())

# 各食事の入力欄を追加
if "records" not in st.session_state:
    st.session_state.records = {}

if record_date not in st.session_state.records:
    st.session_state.records[record_date] = {
        "breakfast": {"content": "", "calories": 0},
        "lunch": {"content": "", "calories": 0},
        "dinner": {"content": "", "calories": 0},
        "snacks": {"content": "", "calories": 0},
        "total_intake": 0,
        "exercise": {"content": "", "calories": 0},
        "calorie_balance": 0
    }

# 朝食
breakfast = st.text_input("朝食（例：トースト1枚、卵1個）", key=f"breakfast_input_{record_date}")
if st.button("朝食を確定", key=f"confirm_breakfast_{record_date}"):
    breakfast_data = get_calories_from_chatgpt_with_llmchain(breakfast)
    st.session_state.records[record_date]["breakfast"]["content"] = breakfast
    st.session_state.records[record_date]["breakfast"]["calories"] = extract_calories(breakfast_data["total_calories"])
st.write(f"朝食のカロリー: {st.session_state.records[record_date]['breakfast']['calories']} kcal")

# 昼食
lunch = st.text_input("昼食（例：カレーライス）", key=f"lunch_input_{record_date}")
if st.button("昼食を確定", key=f"confirm_lunch_{record_date}"):
    lunch_data = get_calories_from_chatgpt_with_llmchain(lunch)
    st.session_state.records[record_date]["lunch"]["content"] = lunch
    st.session_state.records[record_date]["lunch"]["calories"] = extract_calories(lunch_data["total_calories"])
st.write(f"昼食のカロリー: {st.session_state.records[record_date]['lunch']['calories']} kcal")

# 夕食
dinner = st.text_input("夕食（例：焼き魚、味噌汁）", key=f"dinner_input_{record_date}")
if st.button("夕食を確定", key=f"confirm_dinner_{record_date}"):
    dinner_data = get_calories_from_chatgpt_with_llmchain(dinner)
    st.session_state.records[record_date]["dinner"]["content"] = dinner
    st.session_state.records[record_date]["dinner"]["calories"] = extract_calories(dinner_data["total_calories"])
st.write(f"夕食のカロリー: {st.session_state.records[record_date]['dinner']['calories']} kcal")

# その他
snacks = st.text_input("その他（例：お菓子、飲み物）", key=f"snacks_input_{record_date}")
if st.button("その他を確定", key=f"confirm_snacks_{record_date}"):
    snacks_data = get_calories_from_chatgpt_with_llmchain(snacks)
    st.session_state.records[record_date]["snacks"]["content"] = snacks
    st.session_state.records[record_date]["snacks"]["calories"] = extract_calories(snacks_data["total_calories"])
st.write(f"その他のカロリー: {st.session_state.records[record_date]['snacks']['calories']} kcal")

# 摂取カロリーを自動計算
total_intake = (
    st.session_state.records[record_date]["breakfast"]["calories"] +
    st.session_state.records[record_date]["lunch"]["calories"] +
    st.session_state.records[record_date]["dinner"]["calories"] +
    st.session_state.records[record_date]["snacks"]["calories"]
)
st.session_state.records[record_date]["total_intake"] = total_intake
st.write(f"摂取カロリーの合計: {total_intake} kcal")

# 修正後の運動内容の入力と消費カロリー計算
exercise = st.text_input("運動内容（例：ウォーキング30分）", key=f"exercise_input_{record_date}")
if st.button("運動を確定", key=f"confirm_exercise_{record_date}"):
    exercise_data = get_exercise_calories_from_chatgpt_with_llmchain(exercise)
    st.session_state.records[record_date]["exercise"]["content"] = exercise
    st.session_state.records[record_date]["exercise"]["calories"] = extract_calories(exercise_data["total_calories"])
st.write(f"運動による消費カロリー: {st.session_state.records[record_date]['exercise']['calories']} kcal")

# カロリー収支の計算
calorie_balance = total_intake - bmr - st.session_state.records[record_date]["exercise"]["calories"]
st.session_state.records[record_date]["calorie_balance"] = calorie_balance
if calorie_balance > 0:
    st.write(f"カロリー収支: +{calorie_balance:.2f} kcal（摂取超過）")
else:
    st.write(f"カロリー収支: {calorie_balance:.2f} kcal（消費超過）")

if st.button("記録を保存", key=f"save_record_{record_date}"):
    st.success(f"{record_date} の記録を保存しました。")

# 表形式で目標と実績を表示
st.header("📊 目標と実績")

# 表データの準備
if "graph_data" not in st.session_state:
    st.session_state.graph_data = {
        "日付": [],
        "目標 (kcal)": [],
        "実績 (kcal)": []
    }

# 実績データを更新
if st.button("表を更新"):
    st.session_state.graph_data["日付"].append(record_date)
    st.session_state.graph_data["目標 (kcal)"].append(bmr)  # 目標値としてBMRを使用
    st.session_state.graph_data["実績 (kcal)"].append(total_intake)

# 表の描画
st.table(st.session_state.graph_data)

from langchain.prompts import PromptTemplate

# プロのトレーナーAI用のプロンプトテンプレートを作成
trainer_prompt_template = PromptTemplate(
    input_variables=["history", "user_input"],
    template=(
        "あなたはプロのトレーナーです。以下の会話履歴を基に、ユーザーの健康やダイエットに関する質問に答えてください。\n\n"
        "会話履歴:\n"
        "{history}\n\n"
        "ユーザーの質問:\n"
        "{user_input}\n\n"
        "プロのトレーナーとして、具体的で役立つアドバイスを提供してください。"
    )
)

# サイドバーにプロのトレーナーとの対話型AIを追加
st.sidebar.header("🤖 プロのトレーナーAIに相談")
if "trainer_ai_history" not in st.session_state:
    st.session_state.trainer_ai_history = []  # 会話履歴を初期化

# ユーザー入力
user_question = st.sidebar.text_input("トレーナーに相談する内容を入力してください：", key="trainer_ai_input")
if st.sidebar.button("相談する", key="trainer_ai_button"):
    if user_question.strip():
        # 会話履歴にユーザーの質問を追加
        st.session_state.trainer_ai_history.append({"role": "user", "content": user_question})
        
        # 会話履歴を文字列形式に変換
        history = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.trainer_ai_history]
        )
        
        # プロンプトを生成
        prompt = trainer_prompt_template.format(history=history, user_input=user_question)
        
        # LLMにプロンプトを渡して応答を取得
        trainer_response = llm.predict(prompt)
        
        # 会話履歴にAIの応答を追加
        st.session_state.trainer_ai_history.append({"role": "assistant", "content": trainer_response})
        
        # AIの応答を表示
        st.sidebar.write(f"**トレーナーAI**: {trainer_response}")
        
