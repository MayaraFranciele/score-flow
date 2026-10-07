PROBLEMAS = {
    "credito": {
        "numericas": {
            "idade": {"min": 18, "max": 80, "default": 35},
            "renda_mensal": {"min": 1200, "max": 20000, "default": 4200},
            "valor_emprestimo": {"min": 500, "max": 120000, "default": 18000},
            "prazo_meses": {"min": 6, "max": 96, "default": 36},
            "score_credito": {"min": 300, "max": 850, "default": 640},
            "tempo_emprego": {"min": 0, "max": 30, "default": 4},
            "divida_total": {"min": 0, "max": 95000, "default": 9000},
            "comprometimento_renda": {"min": 0.0, "max": 1.5, "default": 0.28},
        },
        "categoricas": {
            "finalidade": ["educacao", "veiculo", "casa", "consumo", "negocios", "outros"],
            "posse_imovel": ["sim", "nao"],
        },
    }
}
