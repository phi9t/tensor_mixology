"""FLOPs analysis summary with comprehensive comparisons."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from flops_counter import (
    compare_flops_efficiency,
    count_attention_flops,
    count_rank_mixer_flops,
    count_token_mixer_flops,
    format_flops,
)

# Initialize Rich console
console = Console()


def main():
    """Generate comprehensive FLOPs analysis summary."""
    console.print(
        Panel(
            "[bold blue]FLOPs ANALYSIS SUMMARY[/bold blue]\n"
            "Comprehensive computational analysis of TokenMixer vs Attention",
            title="Analysis Report",
            border_style="blue",
        )
    )

    # Key comparison
    console.print(
        "\n[bold cyan]🔍 KEY COMPARISON (Tokens=8, Hidden=512, Heads=8)[/bold cyan]"
    )
    console.print("─" * 60)

    token_mixer_result = count_token_mixer_flops(8, 512, 8)
    attention_result = count_attention_flops(512, 8, 8)
    rank_mixer_result = count_rank_mixer_flops(8, 512, 8, 2)

    # Create comparison table
    comparison_table = Table(
        title="Module Comparison", show_header=True, header_style="bold magenta"
    )
    comparison_table.add_column("Module", style="cyan", no_wrap=True)
    comparison_table.add_column("FLOPs", style="green", justify="right")
    comparison_table.add_column("Parameters", style="yellow", justify="right")
    comparison_table.add_column("FLOPs/Param", style="blue", justify="right")

    comparison_table.add_row(
        "TokenMixer",
        format_flops(token_mixer_result["flops"]),
        f"{token_mixer_result['params']:,}",
        f"{token_mixer_result['flops_per_param']:.2f}",
    )
    comparison_table.add_row(
        "Attention",
        format_flops(attention_result["flops"]),
        f"{attention_result['params']:,}",
        f"{attention_result['flops_per_param']:.2f}",
    )
    comparison_table.add_row(
        "RankMixer",
        format_flops(rank_mixer_result["flops"]),
        f"{rank_mixer_result['params']:,}",
        f"{rank_mixer_result['flops_per_param']:.2f}",
    )

    console.print(comparison_table)

    # Efficiency analysis
    console.print("\n[bold cyan]📊 EFFICIENCY ANALYSIS[/bold cyan]")
    console.print("─" * 60)

    tm_vs_attn = compare_flops_efficiency(
        token_mixer_result["flops"],
        token_mixer_result["params"],
        attention_result["flops"],
        attention_result["params"],
    )
    rm_vs_attn = compare_flops_efficiency(
        rank_mixer_result["flops"],
        rank_mixer_result["params"],
        attention_result["flops"],
        attention_result["params"],
    )

    efficiency_table = Table(
        title="Efficiency Comparison", show_header=True, header_style="bold magenta"
    )
    efficiency_table.add_column("Comparison", style="cyan", no_wrap=True)
    efficiency_table.add_column("FLOPs Ratio", style="green", justify="right")
    efficiency_table.add_column("Params Ratio", style="yellow", justify="right")
    efficiency_table.add_column("Overall Efficiency", style="blue", justify="right")

    efficiency_table.add_row(
        "TokenMixer vs Attention",
        f"{tm_vs_attn['flops_ratio']:.2f}x",
        f"{tm_vs_attn['params_ratio']:.2f}x",
        f"{1/tm_vs_attn['flops_per_param_ratio']:.0f}x",
    )
    efficiency_table.add_row(
        "RankMixer vs Attention",
        f"{rm_vs_attn['flops_ratio']:.2f}x",
        f"{rm_vs_attn['params_ratio']:.2f}x",
        f"{1/rm_vs_attn['flops_per_param_ratio']:.2f}x",
    )

    console.print(efficiency_table)

    # Scaling analysis
    console.print("\n[bold cyan]📈 SCALING ANALYSIS[/bold cyan]")
    console.print("─" * 60)

    scaling_text = Text()
    scaling_text.append("TokenMixer scaling characteristics:\n", style="bold green")
    scaling_text.append(
        "  ✅ Linear scaling with sequence length: O(T)\n", style="green"
    )
    scaling_text.append(
        "  ✅ Linear scaling with hidden dimension: O(D)\n", style="green"
    )
    scaling_text.append(
        "  ✅ Constant scaling with number of heads: O(1)\n", style="green"
    )
    scaling_text.append(
        "  ✅ Parameter count: O(D) - very efficient!\n\n", style="green"
    )

    scaling_text.append("Attention scaling characteristics:\n", style="bold yellow")
    scaling_text.append(
        "  ⚠️ Quadratic scaling with sequence length: O(T²)\n", style="yellow"
    )
    scaling_text.append(
        "  ✅ Linear scaling with hidden dimension: O(D)\n", style="yellow"
    )
    scaling_text.append(
        "  ✅ Linear scaling with number of heads: O(H)\n", style="yellow"
    )
    scaling_text.append("  ⚠️ Parameter count: O(D²) - less efficient", style="yellow")

    console.print(
        Panel(scaling_text, title="Scaling Characteristics", border_style="cyan")
    )

    # Key insights
    console.print("\n[bold cyan]💡 KEY INSIGHTS[/bold cyan]")
    console.print("─" * 60)

    insights_text = Text()
    insights_text.append("1. COMPUTATIONAL EFFICIENCY:\n", style="bold blue")
    insights_text.append(
        "   • TokenMixer and Attention have similar FLOPs for small sequences\n",
        style="blue",
    )
    insights_text.append(
        "   • TokenMixer scales linearly with sequence length (O(T))\n", style="blue"
    )
    insights_text.append(
        "   • Attention scales quadratically with sequence length (O(T²))\n",
        style="blue",
    )
    insights_text.append(
        "   • For long sequences, TokenMixer becomes significantly more efficient\n\n",
        style="blue",
    )

    insights_text.append("2. PARAMETER EFFICIENCY:\n", style="bold green")
    insights_text.append(
        "   • TokenMixer uses dramatically fewer parameters (1,024 vs 1,050,624)\n",
        style="green",
    )
    insights_text.append(
        "   • TokenMixer is ~1,000x more parameter efficient\n", style="green"
    )
    insights_text.append(
        "   • This makes TokenMixer much more memory efficient\n\n", style="green"
    )

    insights_text.append("3. PRACTICAL IMPLICATIONS:\n", style="bold cyan")
    insights_text.append(
        "   • TokenMixer is ideal for long sequences (documents, code, etc.)\n",
        style="cyan",
    )
    insights_text.append(
        "   • Attention is better for short sequences with complex relationships\n",
        style="cyan",
    )
    insights_text.append(
        "   • RankMixer combines the best of both approaches\n", style="cyan"
    )
    insights_text.append(
        "   • TokenMixer can handle much longer sequences on the same hardware",
        style="cyan",
    )

    console.print(Panel(insights_text, title="Key Insights", border_style="blue"))

    # Performance recommendations
    console.print("\n[bold cyan]🚀 PERFORMANCE RECOMMENDATIONS[/bold cyan]")
    console.print("─" * 60)

    recommendations_text = Text()
    recommendations_text.append("Use TokenMixer when:\n", style="bold green")
    recommendations_text.append("  • Sequence length > 100 tokens\n", style="green")
    recommendations_text.append("  • Memory is constrained\n", style="green")
    recommendations_text.append("  • You need linear scaling\n", style="green")
    recommendations_text.append(
        "  • Processing documents, code, or long texts\n\n", style="green"
    )

    recommendations_text.append("Use Attention when:\n", style="bold yellow")
    recommendations_text.append("  • Sequence length < 100 tokens\n", style="yellow")
    recommendations_text.append(
        "  • Complex token relationships are important\n", style="yellow"
    )
    recommendations_text.append("  • Memory is not a constraint\n", style="yellow")
    recommendations_text.append(
        "  • Processing short sequences with rich interactions\n\n", style="yellow"
    )

    recommendations_text.append("Use RankMixer when:\n", style="bold blue")
    recommendations_text.append(
        "  • You want the benefits of both approaches\n", style="blue"
    )
    recommendations_text.append(
        "  • Building a complete language model\n", style="blue"
    )
    recommendations_text.append(
        "  • Need both local and global token mixing\n", style="blue"
    )
    recommendations_text.append(
        "  • Have sufficient computational resources", style="blue"
    )

    console.print(
        Panel(recommendations_text, title="Recommendations", border_style="green")
    )

    # BERT-like model comparison
    console.print("\n[bold cyan]🔬 BERT-LIKE MODEL COMPARISON (12 layers)[/bold cyan]")
    console.print("─" * 60)

    # Calculate BERT-like model FLOPs
    token_mixer_total_flops = token_mixer_result["flops"] * 12
    token_mixer_total_params = token_mixer_result["params"] * 12
    attention_total_flops = attention_result["flops"] * 12
    attention_total_params = attention_result["params"] * 12
    rank_mixer_bert = count_rank_mixer_flops(8, 512, 8, 12)

    bert_table = Table(
        title="BERT-Like Model Comparison",
        show_header=True,
        header_style="bold magenta",
    )
    bert_table.add_column("Model Type", style="cyan", no_wrap=True)
    bert_table.add_column("Total FLOPs", style="green", justify="right")
    bert_table.add_column("Total Parameters", style="yellow", justify="right")
    bert_table.add_column("FLOPs/Param", style="blue", justify="right")

    bert_table.add_row(
        "TokenMixer BERT",
        format_flops(token_mixer_total_flops),
        f"{token_mixer_total_params:,}",
        f"{token_mixer_total_flops/token_mixer_total_params:.2f}",
    )
    bert_table.add_row(
        "Attention BERT",
        format_flops(attention_total_flops),
        f"{attention_total_params:,}",
        f"{attention_total_flops/attention_total_params:.2f}",
    )
    bert_table.add_row(
        "RankMixer BERT",
        format_flops(rank_mixer_bert["total_flops"]),
        f"{rank_mixer_bert['total_params']:,}",
        f"{rank_mixer_bert['total_flops']/rank_mixer_bert['total_params']:.2f}",
    )

    console.print(bert_table)

    # Conclusion
    conclusion_text = Text()
    conclusion_text.append("CONCLUSION: ", style="bold green")
    conclusion_text.append(
        "TokenMixer offers superior efficiency for long sequences!", style="green"
    )

    console.print(Panel(conclusion_text, title="Conclusion", border_style="green"))


if __name__ == "__main__":
    main()
