"""Генерирует синтетические обращения клиентов в поддержку, привязанные
к реальным customer_id/orders из etl-portfolio (та же БД etl_portfolio) —
не выдуманные customer_id, а реально существующие клиенты и заказы."""
import random
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db import get_engine

random.seed(42)

# Каждый шаблон соответствует одной из тем src/kb.py (кроме off_topic,
# который намеренно не покрыт базой знаний — проверка честного retrieval:
# RAG не обязан "находить" ответ там, где его нет).
TEMPLATES = [
    ("delivery_status", "Где мой заказ #{order_id}? Жду {product} уже больше недели, трек не обновляется."),
    ("return", "Хочу оформить возврат на {product} из заказа #{order_id} — не подошёл, упаковку сохранил."),
    ("defect", "Товар {product} из заказа #{order_id} пришёл с браком, не включается вообще. Что делать?"),
    ("payment", "Можно ли оплатить заказ в рассрочку и как это оформить на сайте?"),
    ("cancel", "Хочу отменить заказ #{order_id}, оформил буквально 10 минут назад по ошибке."),
    ("loyalty", "Сколько бонусов у меня накопилось и можно ли ими оплатить весь следующий заказ?"),
    ("exchange", "Можно ли поменять {product} из заказа #{order_id} на другой цвет той же модели?"),
    ("wholesale", "Интересует оптовая закупка {product} для офиса, от 30 штук. Какие условия и скидка?"),
    ("complaint_rude", "Очень недоволен обслуживанием по заказу #{order_id}, оператор грубо ответил в чате. Разбираюсь третий день без результата!"),
    ("praise", "Спасибо за быструю доставку {product} по заказу #{order_id}, всё отлично, закажу ещё!"),
    ("off_topic", "Подскажите, а вы сотрудничаете с блогерами для обзоров товаров на YouTube?"),
]

MESSAGE_COUNT = 45  # CPU-инференс на слабом железе (~30-40с/сообщение,
# см. README) — компромисс между репрезентативностью выборки и временем прогона


def fetch_customers(engine) -> list[int]:
    df = pd.read_sql("SELECT customer_id FROM stg_customers", engine)
    return df["customer_id"].tolist()


def fetch_orders(engine) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT o.order_id, o.customer_id, p.name AS product_name
        FROM stg_orders o
        JOIN stg_products p ON p.product_id = o.product_id
        """,
        engine,
    )


def generate_messages() -> None:
    engine = get_engine()
    customers = fetch_customers(engine)
    orders = fetch_orders(engine)
    orders_by_customer = orders.groupby("customer_id")

    rows = []
    for _ in range(MESSAGE_COUNT):
        customer_id = random.choice(customers)
        topic, template = random.choice(TEMPLATES)

        if customer_id in orders_by_customer.groups:
            order = orders_by_customer.get_group(customer_id).sample(1).iloc[0]
            order_id, product = order["order_id"], order["product_name"]
        else:
            order_id, product = random.choice(orders["order_id"].tolist()), random.choice(orders["product_name"].tolist())

        text_ = template.format(order_id=order_id, product=product)
        rows.append({"customer_id": customer_id, "message_text": text_, "_topic": topic})

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE triage_results CASCADE"))
        conn.execute(text("TRUNCATE TABLE client_messages RESTART IDENTITY CASCADE"))
        for row in rows:
            conn.execute(
                text("INSERT INTO client_messages (customer_id, message_text) VALUES (:customer_id, :message_text)"),
                {"customer_id": row["customer_id"], "message_text": row["message_text"]},
            )

    topic_counts = pd.Series([r["_topic"] for r in rows]).value_counts()
    print(f"Inserted {len(rows)} client messages")
    print(topic_counts)


if __name__ == "__main__":
    generate_messages()
