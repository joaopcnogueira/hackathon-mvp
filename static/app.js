/**
 * JavaScript principal da aplicação AutoML Platform.
 */

// Função utilitária para fazer requests com tratamento de erro
async function apiRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Erro na requisição');
        }

        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Formatação de números
function formatNumber(num, decimals = 4) {
    if (typeof num !== 'number' || isNaN(num)) return '-';
    return num.toFixed(decimals);
}

// Formatação de porcentagem
function formatPercent(num, decimals = 2) {
    if (typeof num !== 'number' || isNaN(num)) return '-';
    return num.toFixed(decimals) + '%';
}

// Inicialização de tooltips do Bootstrap
document.addEventListener('DOMContentLoaded', function() {
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipTriggerList.forEach(function(tooltipTriggerEl) {
        new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
