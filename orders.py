import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse
from rich.console import Console
from dotenv import load_dotenv
load_dotenv()

console = Console()

class OrderStatus(str, Enum):
    PENDING = 'pending'
    CONFIRMED = 'confirmed'
    SHIPPED = 'shipped'
    DELIVERED = 'delivered'
    CANCELLED = 'cancelled'

class ReturnReason(str, Enum):
    WRONG_SIZE = 'wrong_size'
    WRONG_COLOR = 'wrong_color'
    NOT_AS_DESCRIBED = 'not_as_described'
    CHANGED_MIND = 'changed_mind'
    DAMAGED = 'damaged'

class EscalationReason(str, Enum):
    COMPLEX_REQUEST = 'complex_request'
    CUSTOMER_DISSATISFIED = 'customer_dissatisfied'
    CUSTOMER_REQUEST = 'customer_request'
    CANNOT_RESOLVE_SITUATION = 'cannot_resolve_situation'

class OrderItem(BaseModel):
    product_code: str
    name: str
    size: str
    color: str
    quantity: int
    price: float

class Address(BaseModel):
    street: str
    city: str
    postal_code: str
    country: str

class Order(BaseModel):
    order_id: str
    status: OrderStatus
    items: list[OrderItem]
    total_amount: float
    created_at: datetime
    shipping_address: Address
    
    @property
    def can_modify(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.CONFIRMED]
    
    @property
    def can_return(self) -> bool:
        if self.status != OrderStatus.DELIVERED:
            return False
        return (datetime.now() - self.created_at).days <= 30

class Database:
    def __init__(self):
        self.orders = {
            '001': Order(
                order_id='001',
                status=OrderStatus.CONFIRMED,
                items=[
                    OrderItem(
                        product_code='LJ001',
                        name='Classic Noir Biker Jacket',
                        size='M',
                        color='Black',
                        quantity=1,
                        price=1499
                    )
                ],
                total_amount=1499,
                created_at=datetime.now() - timedelta(minutes=30),
                shipping_address=Address(
                    street='123 Avenue des Champs-Élysées',
                    city='Paris',
                    postal_code='75008',
                    country='France'
                )
            ),
            '002': Order(
                order_id='002',
                status=OrderStatus.DELIVERED,
                items=[
                    OrderItem(
                        product_code='BG001',
                        name='Leather Tote',
                        size='ONE',
                        color='Black',
                        quantity=1,
                        price=899
                    )
                ],
                total_amount=899,
                created_at=datetime.now() - timedelta(days=15),
                shipping_address=Address(
                    street='10 Fifth Avenue',
                    city='New York',
                    postal_code='10011',
                    country='USA'
                )
            )
        }
    
    def update_order_status(self, order_id: str, status: OrderStatus) -> bool:
        if order_id in self.orders:
            self.orders[order_id].status = status
            return True
        return False
    
    def update_shipping_address(self, order_id: str, new_address: Address) -> bool:
        order = self.orders.get(order_id)
        if order and order.can_modify:
            self.orders[order_id].shipping_address = new_address
            return True
        return False

SYSTEM_PROMPT = """\
You are an AI assistant for for the luxury fashion store Maison Noir.
Your role is to help clients with orders, returns, and related inquiries, using the provided tools.

Key Guidelines:
- Orders can only be modified or canceled when PENDING or CONFIRMED.
- Address changes require a valid new address with all fields.
- There is a 30-day return window for delivered items. Items must be in original condition with tags.
- Always maintain a professional, refined communication tone. Be concise in your answers and precise with details.

Escalate to a human for:
- Complex modifications or special requests.
- Dissatisfied customers.
- When the user specifically requests to talk with a human.
- Any situation you cannot fully resolve.
"""

@dataclass
class Deps:
    db: Database

orders_agent = Agent(
    model='openai:gpt-4o',
    deps_type=Deps,
    system_prompt=SYSTEM_PROMPT
)

@orders_agent.tool
async def get_order_details(ctx: RunContext[Deps], order_id: str) -> str:
    """Get the current status and details of an order."""
    order = ctx.deps.db.orders.get(order_id)
    if not order:
        return 'Order not found'
    
    items = '\n'.join(
        f"- {item.name} ({item.size}, {item.color}) €{item.price}"
        for item in order.items
    )
    address = order.shipping_address

    return (
        f'Status: {order.status.value}\n'
        f'Order date: {order.created_at.strftime('%Y-%m-%d %H:%M')}\n'
        f'Items:\n{items}\n'
        f'Total: €{order.total_amount}\n'
        f'Shipping to: {address.street}, {address.city}, {address.postal_code} {address.country}\n'
    )

@orders_agent.tool
async def update_shipping_address(
    ctx: RunContext[Deps],
    order_id: str,
    street: str,
    city: str,
    postal_code: str,
    country: str
) -> str:
    """Update the shipping address for an order if possible."""
    order = ctx.deps.db.orders.get(order_id)
    if not order:
        return 'Order not found'
    
    if not order.can_modify:
        return f'Cannot modify order - current status is {order.status.value}'
    
    new_address = Address(street=street, city=city, postal_code=postal_code, country=country)
    
    if ctx.deps.db.update_shipping_address(order_id, new_address):
        return f'Successfully updated shipping address'
    return 'Failed to update shipping address'
    
@orders_agent.tool
async def cancel_order(ctx: RunContext[Deps], order_id: str) -> str:
    """Cancel an order if possible."""
    order = ctx.deps.db.orders.get(order_id)
    if not order:
        return 'Order not found'
    
    if not order.can_modify:
        return f'Cannot modify order - current status is {order.status.value}'

    if ctx.deps.db.update_order_status(order_id, OrderStatus.CANCELLED):
        return f'Successfully cancelled order'
    return 'Failed to cancel order'

@orders_agent.tool
async def request_return(
    ctx: RunContext[Deps],
    order_id: str,
    reason: ReturnReason
) -> str:
    """Process a return request for an order."""
    order = ctx.deps.db.orders.get(order_id)
    if not order:
        return 'Order not found'
    
    if not order.can_return:
        if order.status != OrderStatus.DELIVERED:
            return f'Cannot return order - current status is {order.status.value}'
        return 'Cannot return order - outside our 30-day return window.'

    return_id = 'RET-' + uuid4().hex[:12]
    
    return (
        f"Return request approved for order {order_id}:\n"
        f"Reason: {reason.value}\n\n"
        f"A return label with ID {return_id} has been emailed to you. Please ship items within 14 days with all original tags attached."
    )

@orders_agent.tool
async def escalate_to_human(
    ctx: RunContext[Deps],
    reason: EscalationReason,
    details: str,
    high_priority: bool = False
) -> str:
    """Escalate the conversation to a human.
    Set high_priority=True for urgent matters or when customer is clearly dissatisfied.
    """
    response_time = '1 hour' if high_priority else '24 hours'
    return f'This matter has been escalated to our support team. We will contact you within {response_time}.'

def print_tool_calls(messages):
    for m in messages:
        for part in m.parts:
            if part.part_kind == 'tool-call':
                console.print(f'TOOL CALL: {part.tool_name}', style='bold red')
                console.print(str(part.args), style='red', end='\n\n')
            elif part.part_kind == 'tool-return':
                console.print(f'TOOL RETURN: {part.tool_name}', style='bold green')
                console.print(part.content, style='green', end='\n\n')

async def run_agent(logging=True):
    db = Database()
    messages = []
    console.print('Welcome to Maison Noir. How may I assist you today?', style='cyan', end = '\n\n')
    while True:
        result_content = ''
        user_message = input()
        print()
        async with orders_agent.run_stream(
            user_message, message_history=messages, deps=Deps(db=db)
        ) as result:
            async for chunk in result.stream_text(delta=True):
                result_content += chunk
                console.print(chunk, style='cyan', end='')
        
        console.print('\n')
        # The final result message will NOT be added to result messages
        # if you use stream_text(delta=True), so we need to add it manually
        new_messages = result.new_messages() + [
            ModelResponse.from_text(content=result_content, timestamp=result.timestamp())
        ]
        messages += new_messages
        if logging:
            print_tool_calls(new_messages)


if __name__ == '__main__':
    asyncio.run(run_agent())
