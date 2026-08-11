import os
import json
import sqlite3
import io
import asyncio
from datetime import datetime

# Bibliotecas do Telegram
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Biblioteca da API do Gemini
from google import genai
from google.genai import types

# Biblioteca para geração de gráficos
import matplotlib
matplotlib.use('Agg')  # Configura para rodar em servidores sem interface gráfica/GUI
import matplotlib.pyplot as plt

from dotenv import load_dotenv

# ==========================================
# 🔑 CONFIGURAÇÃO DAS CHAVES DE API
# ==========================================

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)


# ==========================================
# 🗄️ GERENCIAMENTO DO BANCO DE DADOS (SQLite)
# ==========================================
DB_NAME = "financas.db"

def init_db():
    """Cria a tabela de transações no banco de dados se não existir."""
    conn = sqlite3.connect(DB_NAME)
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
    conn.close()

def salvar_transacao(tipo: str, valor: float, categoria: str, descricao: str):
    """Salva uma nova transação no banco de dados."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO transacoes (tipo, valor, categoria, descricao)
        VALUES (?, ?, ?, ?)
    """, (tipo.lower(), valor, categoria, descricao))
    conn.commit()
    conn.close()

def obter_resumo_mes_atual():
    """Calcula receitas, despesas, saldo e detalhamento por categoria do mês atual."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    mes_atual = datetime.now().strftime('%Y-%m')

    cursor.execute("""
        SELECT tipo, SUM(valor)
        FROM transacoes
        WHERE strftime('%Y-%m', data_registro) = ?
        GROUP BY tipo
    """, (mes_atual,))
    
    totais = dict(cursor.fetchall())
    total_receita = totais.get('receita', 0.0)
    total_despesa = totais.get('despesa', 0.0)
    saldo = total_receita - total_despesa

    cursor.execute("""
        SELECT categoria, SUM(valor)
        FROM transacoes
        WHERE tipo = 'despesa' AND strftime('%Y-%m', data_registro) = ?
        GROUP BY categoria
        ORDER BY SUM(valor) DESC
    """, (mes_atual,))
    
    categorias_gastos = cursor.fetchall()
    conn.close()
    return total_receita, total_despesa, saldo, categorias_gastos

def obter_ultimos_registros(limite=5):
    """Busca as últimas transações cadastradas."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT tipo, valor, categoria, descricao, datetime(data_registro, 'localtime')
        FROM transacoes
        ORDER BY id DESC
        LIMIT ?
    """, (limite,))
    registros = cursor.fetchall()
    conn.close()
    return registros


# ==========================================
# 📈 GERADOR DE GRÁFICOS VISUAIS
# ==========================================
def gerar_grafico_gastos_mes():
    """Busca os gastos do mês no SQLite e gera um gráfico de rosquinha em memória."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    mes_atual = datetime.now().strftime('%Y-%m')

    cursor.execute("""
        SELECT categoria, SUM(valor)
        FROM transacoes
        WHERE tipo = 'despesa' AND strftime('%Y-%m', data_registro) = ?
        GROUP BY categoria
        ORDER BY SUM(valor) DESC
    """, (mes_atual,))
    
    dados = cursor.fetchall()
    conn.close()

    if not dados:
        return None

    categorias = [item[0] for item in dados]
    valores = [item[1] for item in dados]
    total_gasto = sum(valores)

    cores = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#1A535C', '#FF9F1C', '#9B59B6', '#3498DB', '#95A5A6']

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        valores,
        labels=categorias,
        autopct='%1.1f%%',
        startangle=140,
        colors=cores[:len(categorias)],
        pctdistance=0.75,
        textprops=dict(color="black", weight="bold")
    )

    plt.setp(autotexts, size=9, weight="bold", color="white")
    plt.setp(texts, size=10)

    circulo_centro = plt.Circle((0, 0), 0.55, fc='white')
    fig.gca().add_artist(circulo_centro)

    ax.set_title(
        f"📊 Gastos por Categoria ({datetime.now().strftime('%m/%Y')})\nTotal: R$ {total_gasto:.2f}",
        fontsize=13,
        weight="bold",
        pad=20
    )

    buffer_imagem = io.BytesIO()
    plt.savefig(buffer_imagem, format='png', bbox_inches='tight', dpi=150)
    buffer_imagem.seek(0)
    plt.close(fig)
    return buffer_imagem


# ==========================================
# 🤖 INTENÇÃO E PROMPTS COM IA
# ==========================================
SYSTEM_PROMPT = """
Você é um assistente de controle financeiro extremamente preciso.
Classifique a intenção da mensagem enviada pelo usuário em uma destas opções:

1. REGISTRO FINANCEIRO (Texto curto, foto, áudio ou PDF de extrato/fatura com um ou MÚLTIPLOS lançamentos):
Retorne um JSON no formato:
{
    "acao": "registro",
    "transacoes": [
        {
            "tipo": "despesa" ou "receita",
            "valor": float (ex: 45.50),
            "categoria": string (ex: "Alimentação", "Transporte", "Lazer", "Contas Fixas", "Saúde", "Outros"),
            "descricao": string curta descrevendo a transação
        }
    ]
}

2. PEDIDO DE CONSULTA, RELATÓRIO OU GRÁFICO:
Retorne o JSON:
{
    "acao": "relatorio",
    "tipo_relatorio": "grafico" ou "saldo" ou "historico"
}
"""

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Olá! Eu sou seu assistente financeiro inteligente.\n\n"
        "📥 **Para registrar:** Envie texto, áudio, foto de nota fiscal ou PDF.\n"
        "📊 **Para consultar:** Peça em linguagem natural (ex: *'mostre um gráfico com as categorias de gastos'*) ou use os comandos:\n"
        "• /saldo - Resumo financeiro do mês\n"
        "• /grafico - Relatório visual por categoria\n"
        "• /historico - Últimos lançamentos\n"
        "• /zerar - Limpa o banco de dados"
    )

async def comando_saldo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total_rec, total_desp, saldo, categorias = obter_resumo_mes_atual()
    nome_mes = datetime.now().strftime('%m/%Y')
    
    msg = f"📊 **Resumo Financeiro de {nome_mes}**\n\n"
    msg += f"🟢 **Entradas:** R$ {total_rec:.2f}\n"
    msg += f"🔴 **Saídas:** R$ {total_desp:.2f}\n"
    msg += f"💵 **Saldo Atual:** R$ {saldo:.2f}\n\n"
    
    if categorias:
        msg += "🏷️ **Gastos por Categoria:**\n"
        for cat, valor in categorias:
            msg += f"  • {cat}: R$ {valor:.2f}\n"
    else:
        msg += "✨ Nenhum gasto registrado neste mês!"
        
    await update.message.reply_text(msg, parse_mode="Markdown")

async def comando_historico(update: Update, context: ContextTypes.DEFAULT_TYPE):
    registros = obter_ultimos_registros(limite=5)
    if not registros:
        await update.message.reply_text("📂 Nenhum registro encontrado no banco de dados.")
        return
        
    msg = "📜 **Últimos 5 Lançamentos:**\n\n"
    for tipo, valor, cat, desc, data in registros:
        emoji = "🟢" if tipo == "receita" else "🔴"
        msg += f"{emoji} **R$ {valor:.2f}** | {cat}\n"
        msg += f"   📝 {desc or 'Sem descrição'}\n"
        msg += f"   🕒 {data}\n\n"
        
    await update.message.reply_text(msg, parse_mode="Markdown")

async def comando_grafico(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Gerando seu relatório visual...")
    grafico_img = gerar_grafico_gastos_mes()
    
    if not grafico_img:
        await update.message.reply_text("✨ Você ainda não tem despesas registradas neste mês para gerar um gráfico!")
        return

    await update.message.reply_photo(
        photo=grafico_img,
        caption="📈 **Relatório Visual de Despesas por Categoria**\nAcompanhe para onde seu dinheiro está indo neste mês!",
        parse_mode="Markdown"
    )

async def comando_zerar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transacoes;")
    conn.commit()
    conn.close()
    await update.message.reply_text("🧹 **Banco de dados zerado com sucesso!** Todos os lançamentos foram apagados.", parse_mode="Markdown")

async def processar_entrada_financeira(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    await message.reply_text("⏳ Processando seu documento/registro (pode levar alguns segundos)...")
    
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
            prompt_content.append("Extraia os dados financeiros desta foto/recibo.")

        # 3. ÁUDIO
        elif message.voice or message.audio:
            audio_obj = message.voice or message.audio
            audio_file = await audio_obj.get_file()
            file_path = "temp_audio.ogg"
            await audio_file.download_to_drive(file_path)
            uploaded_file = client.files.upload(file=file_path)
            prompt_content.append(uploaded_file)
            prompt_content.append("Ouça o áudio e extraia os dados da transação financeira.")

        # 4. ARQUIVOS PDF (Extratos longos ou comprovantes)
        elif message.document:
            doc = message.document
            file_extension = os.path.splitext(doc.file_name)[1].lower() if doc.file_name else ".pdf"
            file_path = f"temp_doc{file_extension}"
            doc_file = await doc.get_file()
            await doc_file.download_to_drive(file_path)
            
            uploaded_file = client.files.upload(file=file_path)
            prompt_content.append(uploaded_file)
            prompt_content.append(
                "Analise este extrato/documento PDF. "
                "Extraia todas as transações (entradas e saídas) visíveis no documento."
            )

        # Chamada com modelo oficial e retentativa assíncrona
# Tentativas de chamada com tratamento limpo de limites de requisição
        max_tentativas = 3
        response = None

        for tentativa in range(max_tentativas):
            try:
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=prompt_content,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json"
                    )
                )
                break  # Sucesso! Sai do loop
            except Exception as err:
                erro_str = str(err)
                if ("429" in erro_str or "RESOURCE_EXHAUSTED" in erro_str) and tentativa < max_tentativas - 1:
                    # Aguarda 10 segundos para a cota por minuto reiniciar
                    await asyncio.sleep(10)
                else:
                    raise err

        if not response or not response.text:
            raise Exception("A IA não retornou uma resposta válida.")

        dados = json.loads(response.text)

        # VERIFICA A AÇÃO SOLICITADA PELA IA
        acao = dados.get("acao", "registro")

        if acao == "relatorio":
            tipo_rel = dados.get("tipo_relatorio", "grafico")
            if tipo_rel == "grafico":
                await comando_grafico(update, context)
            elif tipo_rel == "saldo":
                await comando_saldo(update, context)
            else:
                await comando_historico(update, context)
            return

        # GRAVAÇÃO DAS TRANSAÇÕES (Suporta 1 ou VÁRIAS transações)
        lista_transacoes = dados.get("transacoes", [])
        
        # Fallback de segurança se a IA retornar estrutura simplificada
        if not lista_transacoes and "valor" in dados:
            lista_transacoes = [dados]

        if not lista_transacoes:
            await message.reply_text("⚠️ Nenhuma transação clara foi encontrada no documento.")
            return

        total_salvo = 0
        soma_despesas = 0.0
        soma_receitas = 0.0

        for t in lista_transacoes:
            tipo_bruto = t.get('tipo')
            tipo = str(tipo_bruto).lower() if tipo_bruto else 'despesa'
            valor = float(t.get('valor') or 0.0)
            categoria = str(t.get('categoria') or 'Outros')
            descricao = str(t.get('descricao') or '')

            if valor > 0:
                salvar_transacao(tipo, valor, categoria, descricao)
                total_salvo += 1
                if tipo == 'receita':
                    soma_receitas += valor
                else:
                    soma_despesas += valor

        # Resumo final enviado ao usuário
        if total_salvo == 1:
            texto_resposta = (
                f"✅ **Lançamento Salvo!**\n\n"
                f"💰 **Valor:** R$ {lista_transacoes[0].get('valor', 0.0):.2f}\n"
                f"🏷️ **Categoria:** {lista_transacoes[0].get('categoria', 'Outros')}\n"
                f"📝 **Descrição:** {lista_transacoes[0].get('descricao', '-')}\n\n"
                f"💡 *Digite /saldo para ver o resumo atualizado.*"
            )
        else:
            texto_resposta = (
                f"✅ **Extrato Processado com Sucesso!**\n\n"
                f"📊 **Total de lançamentos lidos:** {total_salvo}\n"
                f"🟢 **Total Entradas:** R$ {soma_receitas:.2f}\n"
                f"🔴 **Total Saídas:** R$ {soma_despesas:.2f}\n\n"
                f"💡 *Digite /saldo para ver o resumo completo ou /grafico para o relatório visual.*"
            )

        await message.reply_text(texto_resposta, parse_mode="Markdown")

    except Exception as e:
        await message.reply_text(f"❌ Erro ao processar o arquivo: {str(e)}")

    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


# ==========================================
# 🚀 INICIALIZAÇÃO DO BOT
# ==========================================
if __name__ == "__main__":
    init_db()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Handlers dos Comandos
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("saldo", comando_saldo))
    app.add_handler(CommandHandler("historico", comando_historico))
    app.add_handler(CommandHandler("grafico", comando_grafico))
    app.add_handler(CommandHandler("relatorio", comando_grafico))
    app.add_handler(CommandHandler("zerar", comando_zerar))

    # Handler para mensagens de Texto, Foto, Áudio e PDFs
    app.add_handler(MessageHandler(
        filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO | filters.Document.ALL,
        processar_entrada_financeira
    ))

    print("🤖 Bot financeiro completo rodando...")
    app.run_polling()
