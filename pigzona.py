import os
import json
import re
import sqlite3
import io
import asyncio
import threading
from datetime import datetime

# Servidor Web para a Render
from flask import Flask

# Bibliotecas do Telegram
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# Biblioteca da API do Gemini
from google import genai
from google.genai import types

# Biblioteca para geração de gráficos
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv


# ==========================================
# CONFIGURAÇÃO DAS CHAVES DE API
# ==========================================

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("A variável TELEGRAM_BOT_TOKEN não foi configurada.")

if not GEMINI_API_KEY:
    raise RuntimeError("A variável GEMINI_API_KEY não foi configurada.")

client = genai.Client(api_key=GEMINI_API_KEY)


# ==========================================
# GERENCIAMENTO DO BANCO DE DADOS (SQLite)
# ==========================================

DB_NAME = "financas.db"


def limpar_json_resposta(texto: str) -> str:
    """Extrai e limpa o bloco JSON retornado pelo modelo."""
    texto = re.sub(r"```json\s*", "", texto, flags=re.IGNORECASE)
    texto = re.sub(r"```\s*$", "", texto)
    texto = texto.strip()

    match = re.search(r"(\{.*\}|\[.*\])", texto, re.DOTALL)
    if match:
        return match.group(0)

    return texto


def init_db():
    """Cria a tabela de transações se ela ainda não existir."""
    conn = sqlite3.connect(DB_NAME)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transacoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tipo TEXT CHECK(tipo IN ('receita', 'despesa')),
                valor REAL NOT NULL,
                categoria TEXT NOT NULL,
                descricao TEXT,
                data_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    finally:
        conn.close()


def salvar_transacao(
    tipo: str,
    valor: float,
    categoria: str,
    descricao: str
):
    """Salva uma nova transação no banco de dados."""
    conn = sqlite3.connect(DB_NAME)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transacoes (tipo, valor, categoria, descricao)
            VALUES (?, ?, ?, ?)
        """, (tipo.lower(), valor, categoria, descricao))
        conn.commit()
    finally:
        conn.close()


def obter_resumo_mes_atual():
    """Calcula receitas, despesas, saldo e gastos por categoria."""
    conn = sqlite3.connect(DB_NAME)
    try:
        cursor = conn.cursor()
        mes_atual = datetime.now().strftime("%Y-%m")

        cursor.execute("""
            SELECT tipo, SUM(valor)
            FROM transacoes
            WHERE strftime('%Y-%m', data_registro) = ?
            GROUP BY tipo
        """, (mes_atual,))

        totais = dict(cursor.fetchall())
        total_receita = float(totais.get("receita", 0.0) or 0.0)
        total_despesa = float(totais.get("despesa", 0.0) or 0.0)
        saldo = total_receita - total_despesa

        cursor.execute("""
            SELECT categoria, SUM(valor)
            FROM transacoes
            WHERE tipo = 'despesa'
              AND strftime('%Y-%m', data_registro) = ?
            GROUP BY categoria
            ORDER BY SUM(valor) DESC
        """, (mes_atual,))

        categorias_gastos = cursor.fetchall()

        return (
            total_receita,
            total_despesa,
            saldo,
            categorias_gastos,
        )
    finally:
        conn.close()


def obter_ultimos_registros(limite=5):
    """Busca as últimas transações cadastradas."""
    conn = sqlite3.connect(DB_NAME)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tipo, valor, categoria, descricao,
                   datetime(data_registro, 'localtime')
            FROM transacoes
            ORDER BY id DESC
            LIMIT ?
        """, (limite,))

        return cursor.fetchall()
    finally:
        conn.close()


# ==========================================
# GERADOR DE GRÁFICOS VISUAIS
# ==========================================

def gerar_grafico_gastos_mes():
    """Gera um gráfico de rosquinha com os gastos do mês."""
    conn = sqlite3.connect(DB_NAME)

    try:
        cursor = conn.cursor()
        mes_atual = datetime.now().strftime("%Y-%m")

        cursor.execute("""
            SELECT categoria, SUM(valor)
            FROM transacoes
            WHERE tipo = 'despesa'
              AND strftime('%Y-%m', data_registro) = ?
            GROUP BY categoria
            ORDER BY SUM(valor) DESC
        """, (mes_atual,))

        dados = cursor.fetchall()
    finally:
        conn.close()

    if not dados:
        return None

    categorias = [item[0] for item in dados]
    valores = [float(item[1]) for item in dados]
    total_gasto = sum(valores)

    cores = [
        "#FF6B6B",
        "#4ECDC4",
        "#FFE66D",
        "#1A535C",
        "#FF9F1C",
        "#9B59B6",
        "#3498DB",
        "#95A5A6",
    ]

    # Garante cores suficientes mesmo com muitas categorias.
    if len(categorias) > len(cores):
        cores = (cores * ((len(categorias) // len(cores)) + 1))[:len(categorias)]
    else:
        cores = cores[:len(categorias)]

    fig, ax = plt.subplots(figsize=(6, 6))

    wedges, texts, autotexts = ax.pie(
        valores,
        labels=categorias,
        autopct="%1.1f%%",
        startangle=140,
        colors=cores,
        pctdistance=0.75,
        textprops={"color": "black", "weight": "bold"},
    )

    plt.setp(autotexts, size=9, weight="bold", color="white")
    plt.setp(texts, size=10)

    circulo_centro = plt.Circle((0, 0), 0.55, fc="white")
    fig.gca().add_artist(circulo_centro)

    ax.set_title(
        f"Gastos por Categoria ({datetime.now().strftime('%m/%Y')})\n"
        f"Total: R$ {total_gasto:.2f}",
        fontsize=13,
        weight="bold",
        pad=20,
    )

    buffer_imagem = io.BytesIO()
    plt.savefig(
        buffer_imagem,
        format="png",
        bbox_inches="tight",
        dpi=150,
    )
    buffer_imagem.seek(0)
    plt.close(fig)

    return buffer_imagem


# ==========================================
# INTENÇÃO E PROMPTS COM IA
# ==========================================

SYSTEM_PROMPT = """
Você é o "Porquinho", um assistente de controle financeiro inteligente e amigável.
Classifique a intenção da mensagem enviada pelo usuário em uma das três opções abaixo e retorne APENAS o JSON correspondente:
1. REGISTRO FINANCEIRO (Apenas se houver VALORES NUMÉRICOS/MONETÁRIOS para registrar via texto, foto, áudio ou PDF):
{
   "acao": "registro",
   "transacoes": [
       {
           "tipo": "despesa" ou "receita",
           "valor": float (ex: 45.50),
           "categoria": string (ex: "Alimentação", "Transporte", "Contas Fixas", "Saúde", "Lazer", "Outros"),
           "descricao": string curta descrevendo a transação
       }
   ]
}
2. PEDIDO DE CONSULTA, RELATÓRIO OU GRÁFICO (Ex: "qual meu saldo", "mostre o gráfico", "últimos gastos"):
{
   "acao": "relatorio",
   "tipo_relatorio": "grafico" ou "saldo" ou "historico"
}
3. OUTROS / CONVERSA / INSTRUÇÃO SEM VALOR (Quando a mensagem for um comentário, aviso, pergunta geral ou não contiver valores financeiros para salvar):
{
   "acao": "outros",
   "resposta": "Texto curto e amigável respondendo ao usuário, confirmando o entendimento ou explicando como registrar um gasto com valor."
}
"""


# ==========================================
# COMANDOS DO TELEGRAM
# ==========================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Olá! Eu sou seu assistente financeiro inteligente.\n\n"
        "📥 **Para registrar:** envie texto, áudio, foto de nota fiscal ou PDF.\n"
        "📊 **Para consultar:** peça em linguagem natural "
        "(ex.: *mostre um gráfico com as categorias de gastos*) "
        "ou use os comandos:\n"
        "• /saldo - Resumo financeiro do mês\n"
        "• /grafico - Relatório visual por categoria\n"
        "• /historico - Últimos lançamentos\n"
        "• /zerar - Limpa o banco de dados",
        parse_mode="Markdown",
    )


async def comando_saldo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    total_rec, total_desp, saldo, categorias = obter_resumo_mes_atual()
    nome_mes = datetime.now().strftime("%m/%Y")

    msg = f"📊 **Resumo Financeiro de {nome_mes}**\n\n"
    msg += f"🟢 **Entradas:** R$ {total_rec:.2f}\n"
    msg += f"🔴 **Saídas:** R$ {total_desp:.2f}\n"
    msg += f"💵 **Saldo Atual:** R$ {saldo:.2f}\n\n"

    if categorias:
        msg += "🏷️ **Gastos por Categoria:**\n"
        for cat, valor in categorias:
            msg += f"• {cat}: R$ {valor:.2f}\n"
    else:
        msg += "✨ Nenhum gasto registrado neste mês!"

    await update.message.reply_text(msg, parse_mode="Markdown")


async def comando_historico(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    registros = obter_ultimos_registros(limite=5)

    if not registros:
        await update.message.reply_text(
            "📂 Nenhum registro encontrado no banco de dados."
        )
        return

    msg = "📜 **Últimos 5 Lançamentos:**\n\n"

    for tipo, valor, cat, desc, data in registros:
        emoji = "🟢" if tipo == "receita" else "🔴"
        msg += f"{emoji} **R$ {valor:.2f}** | {cat}\n"
        msg += f"   📝 {desc or 'Sem descrição'}\n"
        msg += f"   🕒 {data}\n\n"

    await update.message.reply_text(msg, parse_mode="Markdown")


async def comando_grafico(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "📊 Gerando seu relatório visual..."
    )

    grafico_img = gerar_grafico_gastos_mes()

    if not grafico_img:
        await update.message.reply_text(
            "✨ Você ainda não tem despesas registradas neste mês "
            "para gerar um gráfico!"
        )
        return

    await update.message.reply_photo(
        photo=grafico_img,
        caption=(
            "📈 **Relatório Visual de Despesas por Categoria**\n"
            "Acompanhe para onde seu dinheiro está indo neste mês!"
        ),
        parse_mode="Markdown",
    )


async def comando_zerar(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    conn = sqlite3.connect(DB_NAME)

    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transacoes")
        conn.commit()
    finally:
        conn.close()

    await update.message.reply_text(
        "🧹 **Banco de dados zerado com sucesso!** "
        "Todos os lançamentos foram apagados.",
        parse_mode="Markdown",
    )


# ==========================================
# PROCESSAMENTO DE ENTRADAS COM IA
# ==========================================

async def processar_entrada_financeira(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    message = update.message

    if not message:
        return

    await message.reply_text(
        "⏳ Processando seu documento/registro "
        "(pode levar alguns segundos)..."
    )

    file_path = None
    prompt_content = []

    try:
        # 1. MENSAGEM DE TEXTO
        if message.text:
            prompt_content.append(message.text)

        # 2. FOTO
        elif message.photo:
            photo_file = await message.photo[-1].get_file()
            file_path = "temp_image.jpg"
            await photo_file.download_to_drive(file_path)

            uploaded_file = client.files.upload(file=file_path)
            prompt_content.append(uploaded_file)
            prompt_content.append(
                "Extraia os dados financeiros desta foto/recibo."
            )

        # 3. ÁUDIO
        elif message.voice or message.audio:
            audio_obj = message.voice or message.audio
            audio_file = await audio_obj.get_file()
            file_path = "temp_audio.ogg"
            await audio_file.download_to_drive(file_path)

            uploaded_file = client.files.upload(file=file_path)
            prompt_content.append(uploaded_file)
            prompt_content.append(
                "Ouça o áudio e extraia os dados da transação financeira."
            )

        # 4. ARQUIVO / PDF
        elif message.document:
            doc = message.document

            file_extension = (
                os.path.splitext(doc.file_name)[1].lower()
                if doc.file_name
                else ".pdf"
            )

            file_path = f"temp_doc{file_extension}"

            doc_file = await doc.get_file()
            await doc_file.download_to_drive(file_path)

            uploaded_file = client.files.upload(file=file_path)
            prompt_content.append(uploaded_file)
            prompt_content.append(
                "Analise este extrato/documento. "
                "Extraia todas as transações "
                "(entradas e saídas) visíveis no documento."
            )

        else:
            await message.reply_text(
                "⚠️ Não consegui identificar um texto, foto, áudio ou documento."
            )
            return

        # ==========================================
        # CHAMADA AO GEMINI COM RETENTATIVAS
        # ==========================================

        max_tentativas = 3
        response = None

        for tentativa in range(max_tentativas):
            try:
                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=prompt_content,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                    ),
                )
                break

            except Exception as err:
                erro_str = str(err)

                if (
                    ("429" in erro_str or "RESOURCE_EXHAUSTED" in erro_str)
                    and tentativa < max_tentativas - 1
                ):
                    await asyncio.sleep(10)
                else:
                    raise

        if response is None or not response.text:
            raise RuntimeError(
                "A IA não retornou uma resposta válida."
            )

        texto_limpo = limpar_json_resposta(response.text)
        dados = json.loads(texto_limpo)

        # ==========================================
        # VERIFICA A AÇÃO SOLICITADA PELA IA
        # ==========================================

        acao = dados.get("acao", "registro")

        if acao == "outros":
            resposta_ia = dados.get(
                "resposta",
                "Entendido! Quando quiser registrar esse gasto, "
                "basta me mandar o valor.",
            )
            await message.reply_text(resposta_ia)
            return

        if acao == "relatorio":
            tipo_rel = dados.get("tipo_relatorio", "grafico")

            if tipo_rel == "grafico":
                await comando_grafico(update, context)

            elif tipo_rel == "saldo":
                await comando_saldo(update, context)

            else:
                await comando_historico(update, context)

            return

        # ==========================================
        # GRAVAÇÃO DAS TRANSAÇÕES
        # ==========================================

        lista_transacoes = dados.get("transacoes", [])

        # Compatibilidade caso a IA retorne uma única transação
        # diretamente no objeto JSON.
        if not lista_transacoes and "valor" in dados:
            lista_transacoes = [dados]

        if not lista_transacoes:
            await message.reply_text(
                "⚠️ Nenhuma transação clara foi encontrada "
                "no documento."
            )
            return

        total_salvo = 0
        soma_despesas = 0.0
        soma_receitas = 0.0
        transacoes_salvas = []

        for transacao in lista_transacoes:
            if not isinstance(transacao, dict):
                continue

            tipo_bruto = transacao.get("tipo")
            tipo = (
                str(tipo_bruto).lower()
                if tipo_bruto
                else "despesa"
            )

            if tipo not in ("receita", "despesa"):
                tipo = "despesa"

            try:
                valor = float(transacao.get("valor") or 0.0)
            except (TypeError, ValueError):
                continue

            categoria = str(
                transacao.get("categoria") or "Outros"
            )
            descricao = str(
                transacao.get("descricao") or ""
            )

            if valor <= 0:
                continue

            salvar_transacao(
                tipo,
                valor,
                categoria,
                descricao,
            )

            transacoes_salvas.append({
                "tipo": tipo,
                "valor": valor,
                "categoria": categoria,
                "descricao": descricao,
            })

            total_salvo += 1

            if tipo == "receita":
                soma_receitas += valor
            else:
                soma_despesas += valor

        if total_salvo == 0:
            await message.reply_text(
                "⚠️ Não encontrei nenhuma transação válida para salvar."
            )
            return

        # ==========================================
        # RESPOSTA AO USUÁRIO
        # ==========================================

        if total_salvo == 1:
            t = transacoes_salvas[0]

            texto_resposta = (
                "✅ **Lançamento Salvo!**\n\n"
                f"💰 **Valor:** R$ {t['valor']:.2f}\n"
                f"🏷️ **Categoria:** {t['categoria']}\n"
                f"📝 **Descrição:** {t['descricao'] or '-'}\n\n"
                "💡 *Digite /saldo para ver o resumo atualizado.*"
            )

        else:
            texto_resposta = (
                "✅ **Extrato Processado com Sucesso!**\n\n"
                f"📊 **Total de lançamentos lidos:** {total_salvo}\n"
                f"🟢 **Total Entradas:** R$ {soma_receitas:.2f}\n"
                f"🔴 **Total Saídas:** R$ {soma_despesas:.2f}\n\n"
                "💡 *Digite /saldo para ver o resumo completo "
                "ou /grafico para o relatório visual.*"
            )

        await message.reply_text(
            texto_resposta,
            parse_mode="Markdown",
        )

    except json.JSONDecodeError as err:
        await message.reply_text(
            "❌ A IA retornou uma resposta que não pôde ser "
            f"interpretada como JSON: {err}"
        )

    except Exception as err:
        await message.reply_text(
            f"❌ Erro ao processar o arquivo: {err}"
        )

    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


# ==========================================
# SERVIDOR WEB PARA A RENDER
# ==========================================

web_app = Flask(__name__)


@web_app.route("/")
def home():
    return "🤖 Bot Financeiro rodando online!"


def run_web():
    port = int(os.environ.get("PORT", 8080))
    web_app.run(host="0.0.0.0", port=port)


# ==========================================
# INICIALIZAÇÃO DO BOT E DO SERVIDOR
# ==========================================

if __name__ == "__main__":
    init_db()

    # Inicia o servidor Web em uma thread separada.
    threading.Thread(
        target=run_web,
        daemon=True,
    ).start()

    # Inicia o Bot do Telegram.
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Handlers dos comandos.
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("saldo", comando_saldo))
    app.add_handler(CommandHandler("historico", comando_historico))
    app.add_handler(CommandHandler("grafico", comando_grafico))
    app.add_handler(CommandHandler("relatorio", comando_grafico))
    app.add_handler(CommandHandler("zerar", comando_zerar))

    # Handler para mensagens de texto, foto, áudio e documentos.
    app.add_handler(
        MessageHandler(
            filters.TEXT
            | filters.PHOTO
            | filters.VOICE
            | filters.AUDIO
            | filters.Document.ALL,
            processar_entrada_financeira,
        )
    )

    print("🤖 Bot financeiro completo rodando...")
    app.run_polling()
