# Diretrizes de Desenvolvimento

## Princípio Central: Desenvolvimento Incremental

Este projeto segue uma abordagem estritamente incremental, tanto para a implementação da aplicação quanto para a escrita do artigo (TCC). O objetivo é evitar o cenário onde toda a aplicação é construída de uma vez e o artigo é escrito depois como uma reflexão tardia.

## Regras

1. **Um passo de cada vez.** Cada etapa de desenvolvimento deve ser pequena, compreensível e verificável antes de avançar para a próxima.

2. **Implementar → Validar → Documentar → Avançar.** Nenhuma etapa nova começa sem que a anterior esteja funcionando e documentada no `TCC.md`.

3. **O aluno lidera.** Nenhuma decisão arquitetural ou de implementação significativa é tomada sem discussão e acordo explícito. O Kiro sugere, o aluno decide.

4. **Sem pressa.** Velocidade não é métrica de sucesso aqui. Compreensão profunda e qualidade são. Cada componente deve ser entendido antes de ser integrado.

5. **Documentação contínua.** O `TCC.md` é atualizado a cada etapa concluída, registrando:
   - O que foi implementado
   - Decisões tomadas e suas justificativas
   - Dificuldades encontradas
   - Resultados observados

6. **Escopo controlado.** Resistir à tentação de implementar funcionalidades extras ou otimizações prematuras. Foco no que é necessário para a etapa atual.

7. **Progresso visível.** Cada etapa deve produzir algo que funcione e possa ser demonstrado, mesmo que simples.

## Fluxo de Trabalho

```
┌─────────────────┐
│  Discutir etapa  │
└───────┬─────────┘
        ▼
┌─────────────────┐
│  Implementar     │
└───────┬─────────┘
        ▼
┌─────────────────┐
│  Testar/Validar  │
└───────┬─────────┘
        ▼
┌─────────────────┐
│  Documentar no   │
│  TCC.md          │
└───────┬─────────┘
        ▼
┌─────────────────┐
│  Revisar juntos  │
└───────┬─────────┘
        ▼
┌─────────────────┐
│  Próxima etapa   │
└─────────────────┘
```

## O que isso significa na prática para o Kiro

- Não gerar código de múltiplos componentes de uma vez
- Sempre perguntar antes de tomar decisões de design
- Explicar o "porquê" de cada sugestão, não apenas o "como"
- Esperar confirmação antes de avançar
- Sugerir atualizações no TCC.md após cada etapa concluída
