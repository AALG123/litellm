"use client";

import {
  ConfigType,
  GeneralSettingsFieldName,
  useDeleteProxyConfigField,
  useProxyConfig,
} from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";
import {
  StoreRequestInSpendLogsParams,
  useStoreRequestInSpendLogs,
} from "@/app/(dashboard)/hooks/storeRequestInSpendLogs/useStoreRequestInSpendLogs";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { ClockCircleOutlined } from "@ant-design/icons";
import { Button, Card, Form, Input, InputNumber, Skeleton, Space, Switch, Typography } from "antd";
import React, { useMemo } from "react";

const STORE_PROMPTS_FIELD_NAME = "store_prompts_in_spend_logs";

interface OptionalField {
  readonly name: GeneralSettingsFieldName;
  readonly kind: "duration" | "count";
  readonly label: string;
  readonly placeholder: string;
  readonly fallbackTooltip: string;
}

const OPTIONAL_FIELDS: readonly OptionalField[] = [
  {
    name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_RETENTION_PERIOD,
    kind: "duration",
    label: "Maximum Spend Logs Retention Period (Optional)",
    placeholder: "e.g., 7d, 30d",
    fallbackTooltip:
      "Set the maximum retention period for spend logs (e.g., '7d' for 7 days, '30d' for 30 days). Leave empty for no limit.",
  },
  {
    name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_CLEANUP_BATCH_SIZE,
    kind: "count",
    label: "Spend Logs Cleanup Batch Size (Optional)",
    placeholder: "e.g., 1000",
    fallbackTooltip: "Rows deleted per DELETE statement during cleanup. Leave empty to use the default of 1000.",
  },
  {
    name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_CLEANUP_MAX_BATCHES,
    kind: "count",
    label: "Spend Logs Cleanup Max Batches (Optional)",
    placeholder: "e.g., 500",
    fallbackTooltip:
      "Maximum number of DELETE statements run per table per cleanup run. Leave empty to use the default of 500.",
  },
  {
    name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_CLEANUP_RUN_BUDGET,
    kind: "duration",
    label: "Spend Logs Cleanup Run Budget (Optional)",
    placeholder: "e.g., 5m",
    fallbackTooltip:
      "Wall-clock budget for a whole cleanup run, shared across every table it cleans (e.g., '5m'). Leave empty to use the default of 5m.",
  },
  {
    name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_CLEANUP_BATCH_TIMEOUT,
    kind: "duration",
    label: "Spend Logs Cleanup Batch Timeout (Optional)",
    placeholder: "e.g., 30s",
    fallbackTooltip:
      "Postgres statement and lock timeout applied to each cleanup batch, so cleanup never monopolizes a connection (e.g., '30s'). Leave empty to use the default of 30s.",
  },
];

interface LoggingSettingsFormValues {
  store_prompts_in_spend_logs: boolean;
  maximum_spend_logs_retention_period?: string | null;
  maximum_spend_logs_cleanup_batch_size?: number | null;
  maximum_spend_logs_cleanup_max_batches?: number | null;
  maximum_spend_logs_cleanup_run_budget?: string | null;
  maximum_spend_logs_cleanup_batch_timeout?: string | null;
}

const hasDuration = (value: string | null | undefined): value is string =>
  typeof value === "string" && value.trim() !== "";

const hasCount = (value: number | null | undefined): value is number =>
  typeof value === "number" && Number.isFinite(value);

const buildUpdateParams = (formValues: LoggingSettingsFormValues): StoreRequestInSpendLogsParams => ({
  store_prompts_in_spend_logs: formValues.store_prompts_in_spend_logs,
  ...(hasDuration(formValues.maximum_spend_logs_retention_period) && {
    maximum_spend_logs_retention_period: formValues.maximum_spend_logs_retention_period,
  }),
  ...(hasCount(formValues.maximum_spend_logs_cleanup_batch_size) && {
    maximum_spend_logs_cleanup_batch_size: formValues.maximum_spend_logs_cleanup_batch_size,
  }),
  ...(hasCount(formValues.maximum_spend_logs_cleanup_max_batches) && {
    maximum_spend_logs_cleanup_max_batches: formValues.maximum_spend_logs_cleanup_max_batches,
  }),
  ...(hasDuration(formValues.maximum_spend_logs_cleanup_run_budget) && {
    maximum_spend_logs_cleanup_run_budget: formValues.maximum_spend_logs_cleanup_run_budget,
  }),
  ...(hasDuration(formValues.maximum_spend_logs_cleanup_batch_timeout) && {
    maximum_spend_logs_cleanup_batch_timeout: formValues.maximum_spend_logs_cleanup_batch_timeout,
  }),
});

const omittedFieldNames = (updateParams: StoreRequestInSpendLogsParams): readonly GeneralSettingsFieldName[] =>
  OPTIONAL_FIELDS.map((field) => field.name).filter((name) => !(name in updateParams));

const LoggingSettings: React.FC = () => {
  const [form] = Form.useForm<LoggingSettingsFormValues>();
  const { mutate, isPending } = useStoreRequestInSpendLogs();
  const { mutate: deleteField, isPending: isDeletingField } = useDeleteProxyConfigField();
  const { data: proxyConfigData, isLoading: isLoadingConfig } = useProxyConfig(ConfigType.GENERAL_SETTINGS);

  const describeField = (name: string, fallback: string) =>
    proxyConfigData?.find((field) => field.field_name === name)?.field_description || fallback;

  const initialValues = useMemo(() => {
    const storedValue = (name: string) => proxyConfigData?.find((field) => field.field_name === name)?.field_value;

    return {
      store_prompts_in_spend_logs: storedValue(STORE_PROMPTS_FIELD_NAME) ?? false,
      ...Object.fromEntries(OPTIONAL_FIELDS.map((field) => [field.name, storedValue(field.name)])),
    };
  }, [proxyConfigData]);

  const clearStoredField = (fieldName: GeneralSettingsFieldName) =>
    new Promise<void>((resolve) =>
      deleteField(
        { config_type: ConfigType.GENERAL_SETTINGS, field_name: fieldName },
        {
          onError: (deleteError) => console.warn(`Failed to delete ${fieldName} field (may not exist):`, deleteError),
          onSettled: () => resolve(),
        },
      ),
    );

  const handleFormSubmit = (formValues: LoggingSettingsFormValues) => {
    const updateParams = buildUpdateParams(formValues);
    const submitUpdate = () =>
      mutate(updateParams, {
        onSuccess: () => NotificationsManager.success("Spend logs settings updated successfully"),
        onError: (error) =>
          NotificationsManager.fromBackend("Failed to save spend logs settings: " + parseErrorMessage(error)),
      });

    const fieldsToClear = omittedFieldNames(updateParams);
    if (fieldsToClear.length === 0) {
      submitUpdate();
      return;
    }

    void Promise.all(fieldsToClear.map(clearStoredField)).then(submitUpdate);
  };

  return (
    <Card title="Logging Settings">
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        <Typography.Paragraph style={{ marginBottom: 0 }} type="secondary">
          Proxy-wide settings that control how request and response data are written to spend logs.
        </Typography.Paragraph>

        {isLoadingConfig ? (
          <Skeleton active paragraph={{ rows: 4 }} />
        ) : (
          <Form form={form} layout="vertical" onFinish={handleFormSubmit} initialValues={initialValues}>
            <Form.Item
              label="Store Prompts in Spend Logs"
              name={STORE_PROMPTS_FIELD_NAME}
              tooltip={describeField(
                STORE_PROMPTS_FIELD_NAME,
                "When enabled, prompts will be stored in spend logs for tracking and analysis purposes.",
              )}
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>

            {OPTIONAL_FIELDS.map((field) => (
              <Form.Item
                key={field.name}
                label={field.label}
                name={field.name}
                tooltip={describeField(field.name, field.fallbackTooltip)}
              >
                {field.kind === "duration" ? (
                  <Input placeholder={field.placeholder} prefix={<ClockCircleOutlined />} />
                ) : (
                  <InputNumber min={1} precision={0} placeholder={field.placeholder} style={{ width: "100%" }} />
                )}
              </Form.Item>
            ))}

            <Form.Item>
              <Button type="primary" htmlType="submit" loading={isPending || isDeletingField}>
                {isPending || isDeletingField ? "Saving..." : "Save Settings"}
              </Button>
            </Form.Item>
          </Form>
        )}
      </Space>
    </Card>
  );
};

export default LoggingSettings;
