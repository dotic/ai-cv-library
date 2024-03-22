import configConventional from '@commitlint/config-conventional';

export default {
    extends: ['@commitlint/config-conventional'],
    rules: {
        'body-max-line-length': [0],
        'type-enum': [
            configConventional.rules['type-enum'][0],
            configConventional.rules['type-enum'][1],
            [...configConventional.rules['type-enum'][2], 'wip'],
        ],
    },
};
