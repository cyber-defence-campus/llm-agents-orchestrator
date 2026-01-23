#!/usr/bin/env python3
import asyncio
import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.live import Live
from rich.layout import Layout
from rich.markdown import Markdown

# Configuration
BASE_URL = os.getenv("AGENT_MANAGER_URL", "http://localhost:8083")
console = Console()


class AgentClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=10.0)

    async def close(self):
        await self.client.aclose()

    async def create_agent(
        self, name: str, task: str, model: str | None = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/agents/simple"
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        payload = {"name": name, "task": task, "job_id": job_id}
        if model:
            payload["model"] = model
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    async def list_agents(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/agents"
        response = await self.client.get(url)
        response.raise_for_status()
        return response.json().get("agents", [])

    async def get_agent_details(self, agent_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/agents/{agent_id}"
        response = await self.client.get(url)
        if response.status_code == 404:
            return {}
        if response.status_code == 405:
            return {"error": "API_OUTDATED"}
        response.raise_for_status()
        return response.json()

    async def stop_agent(self, agent_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/agents/{agent_id}/stop"
        response = await self.client.post(url)
        response.raise_for_status()
        return response.json()

    async def send_message(self, agent_id: str, message: str) -> Dict[str, Any]:
        url = f"{self.base_url}/agents/{agent_id}/message"
        payload = {"message": message, "sender": "user"}
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def clear_screen():
    console.clear()


def get_status_color(status):
    status = status.lower()
    if status == "running":
        return "green"
    if status == "completed":
        return "blue"
    if status in ["failed", "error", "stopped"]:
        return "red"
    if status in ["waiting", "initializing"]:
        return "yellow"
    return "white"


async def view_graph(client: AgentClient):
    clear_screen()
    console.print(Panel("[bold blue]Agent Graph[/bold blue]", expand=False))

    try:
        agents = await client.list_agents()
    except Exception as e:
        console.print(f"[bold red]Error listing agents:[/bold red] {e}")
        Prompt.ask("Press Enter to return")
        return

    if not agents:
        console.print("[yellow]No active agents found.[/yellow]")
        Prompt.ask("Press Enter to return")
        return

    # Map by ID for easy lookup
    agent_map = {a["agent_id"]: a for a in agents}
    # Find root agents
    roots = [
        a
        for a in agents
        if not a.get("parent_id") or a.get("parent_id") not in agent_map
    ]

    tree = Tree("[bold]Agent Hierarchy[/bold]")

    def add_children(node, parent_id):
        children = [a for a in agents if a.get("parent_id") == parent_id]
        for child in children:
            status = child["status"]
            color = get_status_color(status)
            label = f"[bold]{child['name'] or 'Unnamed'}[/bold] ([cyan]{child['agent_id']}[/cyan]) - [{color}]{status}[/{color}]"
            child_node = node.add(label)
            add_children(child_node, child["agent_id"])

    for root in roots:
        status = root["status"]
        color = get_status_color(status)
        label = f"[bold]{root['name'] or 'Unnamed'}[/bold] ([cyan]{root['agent_id']}[/cyan]) - [{color}]{status}[/{color}]"
        root_node = tree.add(label)
        add_children(root_node, root["agent_id"])

    console.print(tree)
    console.print("\n")
    Prompt.ask("Press Enter to return to menu")


async def view_agent_details(client: AgentClient, agent_id: str):
    while True:
        clear_screen()
        console.print(
            Panel(f"[bold blue]Agent Details: {agent_id}[/bold blue]", expand=False)
        )

        try:
            details = await client.get_agent_details(agent_id)
        except Exception as e:
            console.print(f"[bold red]Error fetching details:[/bold red] {e}")
            Prompt.ask("Press Enter to return")
            return

        if details.get("error") == "API_OUTDATED":
            console.print(
                "[bold red]Error: The Agent Orchestrator service is outdated.[/bold red]"
            )
            console.print(
                "The running service does not support fetching agent details."
            )
            console.print(
                "[yellow]Please restart the service to apply recent updates:[/yellow]"
            )
            console.print("\n    make run\n")
            Prompt.ask("Press Enter to return")
            return

        if not details:
            console.print("[red]Agent not found or unavailable.[/red]")
            Prompt.ask("Press Enter to return")
            return

        # Header Info
        status = details.get("status", "N/A")
        status_color = get_status_color(status)
        console.print(
            f"Name: [bold magenta]{details.get('agent_name', 'N/A')}[/bold magenta]"
        )
        console.print(f"Status: [bold {status_color}]{status}[/bold {status_color}]")
        console.print(f"Task: {details.get('task', 'N/A')}")
        console.print(Panel("", title="Messages", border_style="dim"))

        # Messages
        messages = details.get("messages", [])
        if not messages:
            console.print("[dim]No messages yet.[/dim]")
        else:
            for msg in messages[-10:]:  # Show last 10
                role = msg.get("role", "unknown")
                color = (
                    "cyan"
                    if role == "user"
                    else "green"
                    if role == "assistant"
                    else "yellow"
                )
                content = msg.get("content", "")
                console.print(
                    f"[[bold {color}]{role.upper()}[/bold {color}]]: {content}"
                )
                console.print(f"[dim]{'-'*20}[/dim]")

        console.print("\n[bold]Options:[/bold]")
        console.print("1. Refresh")
        console.print("2. Stop Agent")
        console.print("3. Send Message")
        console.print("4. Back to Menu")

        choice = Prompt.ask("Select option", choices=["1", "2", "3", "4"], default="1")

        if choice == "1":
            continue
        elif choice == "2":
            if Confirm.ask(f"Are you sure you want to stop agent {agent_id}?"):
                try:
                    await client.stop_agent(agent_id)
                    console.print("[bold red]Stop signal sent.[/bold red]")
                    time.sleep(1)
                except Exception as e:
                    console.print(f"[red]Error stopping agent: {e}[/red]")
                    Prompt.ask("Press Enter to continue")
        elif choice == "3":
            message = Prompt.ask("Enter message")
            if message:
                try:
                    await client.send_message(agent_id, message)
                    console.print("[bold green]Message sent![/bold green]")
                    time.sleep(1)
                except Exception as e:
                    console.print(f"[red]Error sending message: {e}[/red]")
                    Prompt.ask("Press Enter to continue")
        elif choice == "4":
            break


async def create_new_agent(client: AgentClient):
    clear_screen()
    console.print(Panel("[bold green]Create New Agent[/bold green]", expand=False))

    name = Prompt.ask("Agent Name", default="MyAgent")
    task = Prompt.ask("Task Description")

    if not task:
        console.print("[red]Task cannot be empty.[/red]")
        time.sleep(1)
        return

    with console.status("[bold green]Creating agent...[/bold green]"):
        try:
            result = await client.create_agent(name, task)
        except Exception as e:
            console.print(f"[bold red]Error creating agent:[/bold red] {e}")
            Prompt.ask("Press Enter to return")
            return

    console.print(
        f"[bold green]Success![/bold green] Agent ID: [cyan]{result.get('agent_id')}[/cyan]"
    )
    Prompt.ask("Press Enter to return")


async def select_agent_menu(client: AgentClient):
    clear_screen()
    console.print(Panel("[bold blue]Select Agent[/bold blue]", expand=False))

    try:
        agents = await client.list_agents()
    except Exception as e:
        console.print(f"[bold red]Error listing agents:[/bold red] {e}")
        Prompt.ask("Press Enter to return")
        return

    if not agents:
        console.print("[yellow]No active agents found.[/yellow]")
        Prompt.ask("Press Enter to return")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Status")

    for idx, agent in enumerate(agents):
        status = agent["status"]
        color = get_status_color(status)
        table.add_row(
            str(idx + 1),
            agent["agent_id"],
            agent.get("name") or "N/A",
            f"[{color}]{status}[/{color}]",
        )

    console.print(table)
    console.print("\n")

    choice = Prompt.ask("Select agent # (or 'q' to cancel)")
    if choice.lower() == "q":
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(agents):
            agent_id = agents[idx]["agent_id"]
            await view_agent_details(client, agent_id)
        else:
            console.print("[red]Invalid selection.[/red]")
            time.sleep(1)
    except ValueError:
        console.print("[red]Invalid input.[/red]")
        time.sleep(1)


async def main_menu():
    client = AgentClient(BASE_URL)
    try:
        while True:
            clear_screen()
            console.print(
                Panel(
                    "[bold magenta]Agent Orchestrator CLI[/bold magenta]", expand=False
                )
            )
            console.print("1. Create Agent")
            console.print("2. View Agent Graph")
            console.print("3. Select Agent (Details/Messages)")
            console.print("4. Exit")
            console.print("\n")

            choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4"])

            if choice == "1":
                await create_new_agent(client)
            elif choice == "2":
                await view_graph(client)
            elif choice == "3":
                await select_agent_menu(client)
            elif choice == "4":
                console.print("Goodbye!")
                break
    finally:
        await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main_menu())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
        sys.exit(0)
