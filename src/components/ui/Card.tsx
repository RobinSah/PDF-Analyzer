import React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  variant?: 'default' | 'outline' | 'filled';
  title?: string;
  description?: string;
  footer?: React.ReactNode;
}

const Card: React.FC<CardProps> = ({
  children,
  className = '',
  variant = 'default',
  title,
  description,
  footer,
}) => {
  const variantClasses = {
    default: 'bg-white border border-gray-200',
    outline: 'bg-white border border-gray-300',
    filled: 'bg-gray-50 border border-gray-200',
  };

  return (
    <div className={`rounded-lg shadow-sm ${variantClasses[variant]} ${className}`}>
      {(title || description) && (
        <div className="p-5 border-b border-gray-200">
          {title && <h3 className="text-lg font-semibold text-gray-900">{title}</h3>}
          {description && <p className="mt-1 text-sm text-gray-500">{description}</p>}
        </div>
      )}
      <div className="p-5">{children}</div>
      {footer && <div className="px-5 py-4 bg-gray-50 border-t border-gray-200 rounded-b-lg">{footer}</div>}
    </div>
  );
};

export default Card;